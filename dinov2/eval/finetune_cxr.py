# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging, transformers
import os
import sys
from typing import List, Optional
from omegaconf import OmegaConf
from transformers import AutoImageProcessor
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from sklearn.metrics import classification_report
from dinov2.data import SamplerType, make_data_loader, make_cxr_datasets
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate_ft, evaluate_chexpert_test_ft, ValidationTracker,HFModelWithIntermediateLayers
from dinov2.logging import MetricLogger
from torchmetrics.classification import MulticlassAccuracy, MultilabelAUROC, MultilabelAveragePrecision

def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    cfg = OmegaConf.load(args.config_file)
    return cfg

logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of experiemnts to run in parallel to provide mean,std for eval",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use",
        choices=["adamw", "sgd", "adam"],
        default="adamw",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.set_defaults(
        epochs=10,
        batch_size=128,
        num_workers=8,
        epoch_length=1250,
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        val_metric_type=MetricType.MEAN_MULTILABEL_PER_CLASS_AUC_WITH_SIGMOID,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
    )
    return parser


def _load_model(model, path):

    checkpoint = torch.load(path, map_location='cpu') #if distributed.is_main_process() else None
    model.load_state_dict(checkpoint)
    return model

def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [True]: 
            for _lr in learning_rates:
                lr = _lr
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                print(out_dim)
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})
    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifier,
    data_loader,
    metric_type,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    class_mapping=None,
    best_classifier_on_val=None,
    return_preds_gt=False
):
    logger.info("running validation !")
    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)

    postprocessors = linear_classifier
    # metrics = {k: metric.clone() for k in linear_classifier.classifiers_dict}
    metric = metric.clone()

    if data_loader.dataset.dataset_name == 'CheX_Test_Dataset':
        if return_preds_gt:
            _, results_dict_temp , preds, gt = evaluate_chexpert_test_ft(
                feature_model,
                data_loader,
                postprocessors,
                metric,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )
        else:
            _, results_dict_temp = evaluate_chexpert_test_ft(
            feature_model,
            data_loader,
            postprocessors,
            metric,
            torch.cuda.current_device(),
            return_preds_gt=return_preds_gt
            )
    else:
        if return_preds_gt:
            _, results_dict_temp , preds, gt = evaluate_ft(
                feature_model,
                data_loader,
                postprocessors,
                metric,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )
        else:
            _, results_dict_temp = evaluate_ft(
                feature_model,
                data_loader,
                postprocessors,
                metric,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )

    logger.info("")

    results_dict = {}
    max_auc = 0
    best_classifier = ""

    # for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
    #     logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")

    #     if (
    #         best_classifier_on_val is None and metric["macro"].item() > max_auc
    #     ) or classifier_string == best_classifier_on_val:
    #         max_auc = metric["macro"].item()
    #         try:
    #             per_class_auc = metric['per_class'].tolist()
    #         except:
    #             per_class_auc = None
    #         best_classifier = classifier_string
    try:
        per_class_auc = results_dict_temp['per_class'].tolist()
    except:
        per_class_auc = None
    max_auc = results_dict_temp['macro'].item()

    per_class_auc_labels = {k: v for k, v in zip(data_loader.dataset.dataset.pathologies, per_class_auc)} if per_class_auc is not None else None
    results_dict["best_classifier"] = {"name": best_classifier, "macro": max_auc, 'per_class': per_class_auc_labels,'labels': data_loader.dataset.dataset.pathologies}
    logger.info(f"best classifier: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")
    if return_preds_gt:
        return results_dict, preds, gt
    return results_dict


def eval_linear(
    *,
    feature_model,
    linear_classifier, # linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of iter, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric_type,
    training_num_classes,
    resume=True,
    classifier_fpath=None,
    val_class_mapping=None,
):  

    
    checkpointer = Checkpointer(linear_classifier, output_dir, optimizer=optimizer, scheduler=scheduler)

    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    
    validation_tracker = ValidationTracker()
    # periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter) 

    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    for batch in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data, labels = batch['image'], batch['lab']

        if isinstance(data, transformers.image_processing_utils.BatchFeature):
            data = data['pixel_values']
            if len(data.shape) == 5 and data.shape[1] == 1:
                data = data.squeeze(1)

        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifier(features)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item()) 
            metric_logger.update(lr=optimizer.param_groups[0]["lr"]) 
            print("lr", optimizer.param_groups[0]["lr"]) 

        # if iteration - start_iter > 5:
        #     if iteration % running_checkpoint_period == 0:
        #         torch.cuda.synchronize()
        #         if distributed.is_main_process():
        #             logger.info("Checkpointing running_checkpoint")
        #             periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)
        #         torch.cuda.synchronize()
        # periodic_checkpointer.step(iteration)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            torch.cuda.synchronize()
            val_results_dict = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifier=remove_ddp_wrapper(linear_classifier),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,
            )

            val_macro_score = val_results_dict["best_classifier"]["macro"]
            if val_macro_score > validation_tracker.get_best_score():
                if distributed.is_main_process():
                    torch.save(feature_model.state_dict(),os.path.join(checkpointer.save_dir,f"best_feature_model.pth"))
                    torch.save(linear_classifier.state_dict(),os.path.join(checkpointer.save_dir,f"best_model.pth"))
            validation_tracker.update(val_macro_score, iteration,val_results_dict)

        iteration = iteration + 1

    # validate last iteration
    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifier=remove_ddp_wrapper(linear_classifier),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
        class_mapping=val_class_mapping,
    )

    val_macro_score = val_results_dict["best_classifier"]["macro"]
    if val_macro_score > validation_tracker.get_best_score():
        if distributed.is_main_process():
            torch.save(feature_model.state_dict(),os.path.join(checkpointer.save_dir,f"best_feature_model.pth"))
            torch.save(linear_classifier.state_dict(),os.path.join(checkpointer.save_dir,f"best_model.pth"))
    validation_tracker.update(val_macro_score, iteration,val_results_dict)
    
    logger.info(f"Best validation score: {validation_tracker.get_best_score()} from iteration {validation_tracker.get_best_iteration()}, best classifier: {validation_tracker.get_best_score_dict()['best_classifier']}")
    
    best_val_result = validation_tracker.get_best_score_dict() 
    
    torch.distributed.barrier()

    linear_classifier = _load_model(model=linear_classifier, path=os.path.join(checkpointer.save_dir,f"best_model.pth"))

    
    return best_val_result, feature_model, linear_classifier, iteration


def make_eval_data_loader(dataset_configs, batch_size, num_workers, metric_type):
    test_dataset = make_cxr_datasets(
        dataset_configs=dataset_configs,
        transforms=make_classification_eval_transform(resize_size=518,crop_size=518),
    )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
    )
    return test_data_loader


def test_on_datasets(
    feature_model,
    linear_classifiers,
    test_dataset_config,
    batch_size,
    num_workers,
    test_metric_types,
    metrics_file_path,
    training_num_classes,
    iteration,
    best_classifier_on_val,
    prefixstring="",
    test_class_mappings=[None],
):
    results_dict = {}

    for config, class_mapping, metric_type in zip(test_dataset_config, test_class_mappings, test_metric_types):

        test_data_loader = make_eval_data_loader(config, batch_size, num_workers, metric_type)
        results = evaluate_linear_classifiers(
            feature_model,
            remove_ddp_wrapper(linear_classifiers),
            test_data_loader,
            metric_type,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring="",
            class_mapping=class_mapping,
            best_classifier_on_val=best_classifier_on_val,
            return_preds_gt=True
        )
        dataset_results_dict, preds, gt = results


        if test_data_loader.dataset.dataset_name == 'CheX_Test_Dataset':
            results_dict[f"TEST_{config.dataset_name}_macro_5_findings"] = np.mean([dataset_results_dict["best_classifier"]["per_class"][k] for k in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion']])
        
        results_dict[f"TEST_{config.dataset_name}_macro"] = 100.0 * dataset_results_dict["best_classifier"]["macro"]
        results_dict[f"TEST_{config.dataset_name}_per_class"] = dataset_results_dict["best_classifier"]["per_class"]
        results_dict[f"TEST_{config.dataset_name}_labels"] = test_data_loader.dataset.dataset.pathologies
        results_dict[f'TEST Metrics'] = test_metric_types
        
        ap_metric = MultilabelAveragePrecision(task='multilabel', num_labels=gt.shape[-1], average=None)
    
        ap = ap_metric(torch.from_numpy(preds), torch.from_numpy(gt))
        
        hard_preds = (preds > 0.5).astype(int)

        cls_report = classification_report(gt, hard_preds, target_names=test_data_loader.dataset.dataset.pathologies, output_dict=True)
        misc_test_results_path = metrics_file_path.replace("results_eval_linear.json", "misc_test_results.txt")

        if distributed.is_main_process():
            with open(misc_test_results_path, "a") as f:
                f.write(f"--MultilabelAveragePrecision--\n")
                #write macro
                f.write(f"  MACRO : {ap.mean().item()}\n")
                for i, label in enumerate(test_data_loader.dataset.dataset.pathologies):
                    f.write(f"  {label} : {ap[i]}\n")

                f.write(f"--Classification Report t=0.5--\n")
                for key, value in cls_report.items():
                    f.write(f"  {key} : {value}\n")
            # save preds and gt for further analysis
            # avoid overwriting the file name, if it exists, append a number to the end 
            i = 0
            while os.path.exists(metrics_file_path.replace("results_eval_linear.json", f"preds_{config.dataset_name}_{i}.npy")):
                i += 1
            np.save(metrics_file_path.replace("results_eval_linear.json", f"preds_{config.dataset_name}_{i}.npy"), preds)
            np.save(metrics_file_path.replace("results_eval_linear.json", f"gt_{config.dataset_name}_{i}.npy"), gt)



    return results_dict


def run_eval_linear(
    model,
    output_dir,
    train_dataset_config,# train_dataset_str,
    val_dataset_config,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rates,
    autocast_dtype,
    test_dataset_config=None,
    resume=True,
    classifier_fpath=None,
    val_class_mapping_fpath=None,
    test_class_mapping_fpaths=[None],
    val_metric_type=MetricType.MEAN_ACCURACY,
    test_metric_types=None,
    optimizer_str='sgd',
    num_experiments=5,
):
    seed = 0

    if test_dataset_config is None:
        test_dataset_config = [val_dataset_config]
        logger.info('Using validation set as test set!')
    if test_metric_types is None:
        test_metric_types = [val_metric_type] * len(test_dataset_config)
    else:
        assert len(test_metric_types) == len(test_dataset_config)
    assert len(test_dataset_config) == len(test_class_mapping_fpaths)

    train_transform = make_classification_train_transform(crop_size=518,)

    if isinstance(model, transformers.models.dinov2.modeling_dinov2.Dinov2Model):
        train_transform = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        train_transform = partial(train_transform, return_tensors='pt')
        

    train_dataset = make_cxr_datasets(
        dataset_configs=train_dataset_config,
        transforms=train_transform,
    )
    
    logger.info(f'Train labels : {train_dataset.dataset.pathologies}')
    logger.info(f'Train #samples: {len(train_dataset)}')

    training_num_classes = len(train_dataset.dataset.pathologies)
    sampler_type = SamplerType.SHARDED_INFINITE
    # sampler_type = SamplerType.INFINITE
    
    n_last_blocks_list = [4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    if isinstance(model, transformers.models.dinov2.modeling_dinov2.Dinov2Model):
        feature_model = HFModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx, inference_mode=False)
    else:
        feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx, inference_mode=False)
            
    assert all(param.requires_grad for param in feature_model.parameters())

    sample_input = train_dataset[0]['image']
    sample_input = sample_input.pixel_values.squeeze(0) if isinstance(sample_input, transformers.image_processing_utils.BatchFeature) else sample_input
    sample_output = feature_model(sample_input.unsqueeze(0).cuda())
        
    max_iter = epochs * epoch_length

    assert num_experiments ==1, "Only 1 experiment supported for now!"
    assert len(learning_rates) == 1, "Only 1 learning rate supported for now!"
    learning_rate = learning_rates[0]

    out_dim = out_dim = create_linear_input(sample_output, use_n_blocks=n_last_blocks_list[0], use_avgpool=True).shape[1]
    linear_classifier = LinearClassifier(out_dim, use_n_blocks=n_last_blocks_list[0], use_avgpool=True, num_classes=training_num_classes).to(torch.cuda.current_device())
    if distributed.is_enabled():
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier)

    # combine the feature model and the linear classifiers in single model
    params = list(feature_model.parameters()) + list(linear_classifier.parameters())

    if optimizer_str == 'adamw':
        optimizer = torch.optim.AdamW(params,lr=learning_rate, weight_decay=0)
    elif optimizer_str == 'adam':
        optimizer = torch.optim.Adam(params,lr=learning_rate,weight_decay=0)
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(params,lr=learning_rate, momentum=0.9, weight_decay=0)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_str}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    
    checkpointer = Checkpointer(linear_classifier, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    
    val_data_loader = make_eval_data_loader(val_dataset_config, batch_size, 0, val_metric_type) # number of workers=0

    checkpoint_period = save_checkpoint_frequency * epoch_length
    if val_class_mapping_fpath is not None:
        logger.info(f"Using class mapping from {val_class_mapping_fpath}")
        val_class_mapping = np.load(val_class_mapping_fpath)
    else:
        val_class_mapping = None

    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    
    # Train and Val loop
    val_results_dict, feature_model, linear_classifier, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifier=linear_classifier,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=epoch_length,
        eval_period=eval_period_iterations,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        resume=resume,
        val_class_mapping=val_class_mapping,
        classifier_fpath=classifier_fpath,
    )
    
    results_dict = {}
    
    results_dict = test_on_datasets(
        feature_model,
        linear_classifier,
        test_dataset_config,
        batch_size,
        0,  # num_workers,
        test_metric_types,
        metrics_file_path,
        training_num_classes,
        iteration,
        val_results_dict["best_classifier"],
        prefixstring="",
        test_class_mappings=test_class_mappings,
    )

    logger.info("Test Results Dict " + str(results_dict))

    return results_dict


def main(args):

    model, autocast_dtype = setup_and_build_model(args)
    cfg = get_cfg_from_args(args) # YAML configs
    optimizer_str = args.optimizer 


    run_eval_linear(
        model=model,
        output_dir=args.output_dir,
        train_dataset_config=cfg.train.datasets,
        val_dataset_config=cfg.val.datasets,
        test_dataset_config=cfg.test.datasets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
        test_metric_types=args.test_metric_types,
        val_class_mapping_fpath=args.val_class_mapping_fpath,
        test_class_mapping_fpaths=args.test_class_mapping_fpaths,
        optimizer_str=optimizer_str,
        num_experiments=args.num_experiments,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 CXR linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
