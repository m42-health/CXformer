# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch, os
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from collections import defaultdict


logger = logging.getLogger("dinov2")


class ValidationTracker:
    def __init__(self):
        self.best_score = float('-inf')
        self.best_iteration = None
        self.all_scores = []
        self.all_iterations = []
        self.best_score_dict = None

    def update(self, score, iteration, result_dict):
        self.all_scores.append(score)
        self.all_iterations.append(iteration)

        if score > self.best_score:
            self.best_score = score
            self.best_iteration = iteration
            self.best_score_dict = result_dict

    def get_best_score(self):
        return self.best_score

    def get_best_score_dict(self):
        return self.best_score_dict

    def get_best_iteration(self):
        return self.best_iteration
    
    def get_running_summary(self):
        # return dict with score for each iteration number
        return dict(zip(self.all_iterations, self.all_scores))

class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class HFModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx, inference_mode=True):
        super().__init__()
        self.feature_model = feature_model
        if inference_mode:
            self.feature_model.eval()
        else:
            self.feature_model.train()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        self.inference_mode = inference_mode

    def forward(self, images):
        outputs = []
        with torch.inference_mode(mode=self.inference_mode):
            with self.autocast_ctx():
                features = self.feature_model(images,output_hidden_states=True)
                hidden_states = features.hidden_states
                blocks_to_take = range(len(hidden_states) - self.n_last_blocks, len(hidden_states))
                for i in blocks_to_take:
                    h = hidden_states[i] 
                    h = self.feature_model.layernorm(h)
                    outputs.append(h)
                class_tokens =[out[:, 0] for out in outputs] 
        return tuple(zip(outputs, class_tokens))


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx,inference_mode=True):
        super().__init__()
        self.feature_model = feature_model
        if inference_mode:
            self.feature_model.eval()
        else:
            self.feature_model.train()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        self.inference_mode = inference_mode

    def forward(self, images):
        with torch.inference_mode(mode=self.inference_mode):
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features
    
class ModelWithIntermediateLayersForFinetuning(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(
                images, self.n_last_blocks, return_class_token=True
        )
        return features

@torch.inference_mode()
def evaluate_chexpert_test_ft(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds_gt=False
):
    
    patient_outputs = {k: defaultdict(list) for k in metrics.keys()}
    patient_targets = {k: defaultdict(list) for k in metrics.keys()}

    model.eval()
    postprocessors.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch['image']
        targets = batch['lab']
        outputs = model(samples.to(device))
        targets = targets.to(device)
        paths = batch['path']  # Assuming batch['path'] contains the paths to the data
        batch_patient_ids = [next(part for part in p.split(os.sep) if part.startswith('patient')) for p in paths]

        for k, metric in metrics.items():
            logits = postprocessors(outputs) # postprocessor is a linear classifier on top of the ViT backbone

            for pid, pr, tg in zip(batch_patient_ids, logits, targets):
                patient_outputs[k][pid].append(pr)
                patient_targets[k][pid] = tg  # no need to append, the target is the same for all images of the same patient

    # Compute the maximum score for each label for each patient
    for k in metrics.keys():
        for patient_id, outputs in patient_outputs[k].items():
            patient_outputs[k][patient_id] = torch.stack(outputs).max(dim=0)[0]
    for k, metric in metrics.items():
        for patient_id, outputs in patient_outputs[k].items():
            metric.update(preds=outputs.unsqueeze(0), target=patient_targets[k][patient_id].unsqueeze(0))

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    preds, gt = metric.preds, metric.target
    # list of tensors, convert to single tensor
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    # convert to numpy, cpu
    preds = preds.cpu().numpy()
    gt = gt.cpu().numpy()

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_preds_gt:
        return metric_logger_stats, stats, preds, gt
    return metric_logger_stats, stats

@torch.inference_mode()
def evaluate_chexpert_test(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds_gt=False
):
    
    patient_outputs = {k: defaultdict(list) for k in metrics.keys()}
    patient_targets = {k: defaultdict(list) for k in metrics.keys()}

    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch['image']
        targets = batch['lab']
        outputs = model(samples.to(device))
        targets = targets.to(device)
        paths = batch['path']  # Assuming batch['path'] contains the paths to the data
        batch_patient_ids = [next(part for part in p.split(os.sep) if part.startswith('patient')) for p in paths]

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets) # postprocessor is a linear classifier on top of the ViT backbone
            # metric inputs is a dictionary with keys 'outputs' (logits) and 'targets'
            for pid, pr, tg in zip(batch_patient_ids, metric_inputs['preds'], metric_inputs['target']):
                patient_outputs[k][pid].append(pr)
                patient_targets[k][pid] = tg  # no need to append, the target is the same for all images of the same patient

    # Compute the maximum score for each label for each patient
    for k in metrics.keys():
        for patient_id, outputs in patient_outputs[k].items():
            patient_outputs[k][patient_id] = torch.stack(outputs).max(dim=0)[0]
    for k, metric in metrics.items():
        for patient_id, outputs in patient_outputs[k].items():
            metric.update(preds=outputs.unsqueeze(0), target=patient_targets[k][patient_id].unsqueeze(0))

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    preds, gt = metric['macro'].preds, metric['macro'].target
    # list of tensors, convert to single tensor
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    # convert to numpy, cpu
    preds = preds.cpu().numpy()
    gt = gt.cpu().numpy()

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_preds_gt:
        return metric_logger_stats, stats, preds, gt
    return metric_logger_stats, stats

@torch.inference_mode()
def evaluate_ft(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds_gt=False
):
    model.eval()
    postprocessors.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch['image']
        targets = batch['lab']
        with torch.no_grad():
            outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())
        for k, metric in metrics.items():
            with torch.no_grad():
                logits = postprocessors(outputs)
            metric_inputs = {
                "preds":logits,
                "target":targets
            }
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    preds, gt = metric.preds, metric.target
    # list of tensors, convert to single tensor
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    # convert to numpy, cpu
    preds = preds.cpu().numpy()
    gt = gt.cpu().numpy()

    stats = {k: metric.compute() for k, metric in metrics.items()}

    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_preds_gt:
        return metric_logger_stats, stats, preds, gt
    return metric_logger_stats, stats

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds_gt=False
):

    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch['image']
        targets = batch['lab']
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())
        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    preds, gt = metric['macro'].preds, metric['macro'].target
    # list of tensors, convert to single tensor
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    # convert to numpy, cpu
    preds = preds.cpu().numpy()
    gt = gt.cpu().numpy()

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_preds_gt:
        return metric_logger_stats, stats, preds, gt
    return metric_logger_stats, stats


@torch.inference_mode()
def evaluate_knn(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):

    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch['image']
        targets = batch['lab']
        if targets.shape[-1] == 1:
            targets = targets.flatten()
        else:
            raise NotImplementedError("Only single label per image is supported")
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())
        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets.long())
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return metric_logger_stats, stats

def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):

    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets) 
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):

    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None

    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device, dtype=features_rank.dtype)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count # dataset count, num_labels 
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device, dtype=labels_rank.dtype)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all.long(), features_all_ranks)
            all_labels.index_copy_(0, index_all.long(), labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
