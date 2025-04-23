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
from dinov2.eval.setup import setup_and_build_model, setup_and_build_model_raddino
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate, evaluate_chexpert_test, ValidationTracker,HFModelWithIntermediateLayers
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
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
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
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.set_defaults(
        num_workers=8,
        classifier_fpath=None,
        test_class_mapping_fpaths=[None],
        test_metric_types=[MetricType.MEAN_MULTILABEL_PER_CLASS_AUC_WITH_SIGMOID],
    )
    return parser



def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


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

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000,n_layers=1):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes

        self.linear_layers = []
        ####
        self.n_layers = n_layers - 1
        for _ in range(self.n_layers):
            lyr = nn.Linear(self.out_dim, self.out_dim)
            lyr.weight.data.normal_(mean=0.0, std=0.01)
            lyr.bias.data.zero_()
            self.linear_layers.append(lyr)
        self.linear_layers = nn.ModuleList(self.linear_layers)
        #####

        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        if self.n_layers:
            for layer in self.linear_layers:
                output = layer(output)
                output = nn.LeakyReLU(inplace=True)(output)
        return self.linear(output)


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




@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
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
    
    postprocessors = {'classifier': 
                      LinearPostprocessor(linear_classifiers)
                      }
    metrics = {'classifier': metric.clone()}

    if data_loader.dataset.dataset_name == 'CheX_Test_Dataset':
        if return_preds_gt:
            _, results_dict_temp , preds, gt = evaluate_chexpert_test(
                feature_model,
                data_loader,
                postprocessors,
                metrics,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )
        else:
            _, results_dict_temp = evaluate_chexpert_test(
            feature_model,
            data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
            return_preds_gt=return_preds_gt
            )
    else:
        if return_preds_gt:
            _, results_dict_temp , preds, gt = evaluate(
                feature_model,
                data_loader,
                postprocessors,
                metrics,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )
        else:
            _, results_dict_temp = evaluate(
                feature_model,
                data_loader,
                postprocessors,
                metrics,
                torch.cuda.current_device(),
                return_preds_gt=return_preds_gt
            )

    logger.info("")
    results_dict = {}
    max_auc = 0
    best_classifier = ""
    
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric["macro"].item() > max_auc
        ) or classifier_string == best_classifier_on_val:
            max_auc = metric["macro"].item()
            try:
                per_class_auc = metric['per_class'].tolist()
            except:
                per_class_auc = None
            best_classifier = classifier_string

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
        
        try:
            ap_metric = MultilabelAveragePrecision(task='multilabel', num_labels=gt.shape[-1], average=None)
            ap = ap_metric(torch.from_numpy(preds), torch.from_numpy(gt))
            hard_preds = (preds > 0.5).astype(int)
            cls_report = classification_report(gt, hard_preds, target_names=test_data_loader.dataset.dataset.pathologies, output_dict=True)
        except:
            ap = None
            cls_report = None
        
        misc_test_results_path = metrics_file_path.replace("results_eval_linear.json", "misc_test_results.txt")
        if ap is not None and cls_report is not None and distributed.is_main_process():
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
    batch_size,
    output_dir,
    num_workers,
    autocast_dtype,
    test_dataset_config=None,
    resume=True,
    classifier_fpath=None,
    test_class_mapping_fpaths=[None],
    test_metric_types=None,
    optimizer_str='sgd',
    num_experiments=5,
):
    seed = 0
    sd = torch.load(classifier_fpath, map_location='cpu')
    if 'linear_clf' in sd:
        sd = sd['linear_clf']
    
    num_classes = sd['linear.weight'].shape[0]
    
    n_layers = len(sd.keys()) // 2

    linear_classifiers = LinearClassifier(out_dim=sd['linear.weight'].shape[-1], use_n_blocks=4, use_avgpool=True, num_classes=num_classes, n_layers=n_layers)
    msg = linear_classifiers.load_state_dict(sd)
    logger.info(f"Loaded classifier from {classifier_fpath} with msg {msg}")
    linear_classifiers = linear_classifiers.cuda() 

    n_last_blocks = 4
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    if isinstance(model, transformers.models.dinov2.modeling_dinov2.Dinov2Model):
        feature_model = HFModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    else:
        feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    feature_model.eval()
    linear_classifiers.eval()       

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    
    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)


    results_dict = {}

    # Test loop

    results_dict = test_on_datasets(
        feature_model,
        linear_classifiers,
        test_dataset_config,
        batch_size,
        num_workers,  # num_workers,
        test_metric_types,
        metrics_file_path,
        num_classes,
        None, #iteration
        None,
        prefixstring="",
        test_class_mappings=test_class_mappings,
    )


    test_dataset_name = test_dataset_config[0]['dataset_name']

    results_dict["Average_Test_macro"] = np.mean([results_dict[f"TEST_{test_dataset_name}_macro"] ])
    results_dict["STD_Test_macro"] = np.std([results_dict[f"TEST_{test_dataset_name}_macro"] ])
    results_dict["Test_macro_per_head"] = [results_dict[f"TEST_{test_dataset_name}_macro"]]
    logger.info("Test Results Dict " + str(results_dict))


    return results_dict


def main(args):
    import tempfile
    backbone_sd = torch.load(args.pretrained_weights)
    if 'backbone' in backbone_sd.keys():
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            torch.save(backbone_sd['backbone'], temp_file.name)            
            temp_file_path = temp_file.name
            args.pretrained_weights = temp_file_path
            model, autocast_dtype = setup_and_build_model(args)
            # model, autocast_dtype = setup_and_build_model_raddino(args)
    else:
        model, autocast_dtype = setup_and_build_model(args)

    cfg = get_cfg_from_args(args) # YAML configs

    run_eval_linear(
        model=model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        test_dataset_config=cfg.test.datasets,
        num_workers=args.num_workers,
        autocast_dtype=autocast_dtype,
        classifier_fpath=args.classifier_fpath,
        test_metric_types=args.test_metric_types,
        test_class_mapping_fpaths=args.test_class_mapping_fpaths,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 Inference linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
