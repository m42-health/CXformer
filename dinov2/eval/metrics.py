# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MultilabelAUROC, MultilabelAveragePrecision, BinaryAUROC
from torchmetrics.utilities.data import dim_zero_cat, select_topk


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"
    MEAN_MULTILABEL_PER_CLASS_AUC_WITH_SIGMOID = 'mean_multilabel_per_class_auc_with_sigmoid' #per class AUC, macro averaging
    MEAN_MULTILABEL_PER_CLASS_AUPRC_WITH_SIGMOID = 'mean_multilabel_per_class_auprc_with_sigmoid' #per class AUC, macro averaging
    MULTILABEL_PER_CLASS_AUC_WITH_SIGMOID = 'multilabel_per_class_auc_with_sigmoid' # per class AUC, no averaging
    MULTICLASS_TO_BINARY_AUC = 'multiclass_to_binary_auc' 
    MEAN_MULTILABEL_AUC = 'mean_multilabel_auc' # micro AUC

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value



class MultilabelAUROCWithSigmoid(MultilabelAUROC):
    def __init__(self, num_labels, average='macro'):
        super().__init__(num_labels=num_labels, average=average)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Apply sigmoid function to logits before updating internal state
        
        preds = torch.sigmoid(preds)
        super().update(preds, target)

class MultilabelToBinaryAUROC(BinaryAUROC):
    def __init__(self):
        super().__init__()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds will be B,C -> B,
        #apply softmax
        preds = torch.nn.functional.softmax(preds)
        preds = preds[:,0]
        preds = 1 - preds
        target = (target>=1).float()
        super().update(preds, target)

class MultilabelAUPRCWithSigmoid(MultilabelAveragePrecision):
    def __init__(self, num_labels, average='macro'):
        super().__init__(num_labels=num_labels, average=average)
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Apply sigmoid function to logits before updating internal state        
        preds = torch.sigmoid(preds)
        super().update(preds, target)

def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):

    if metric_type == MetricType.MEAN_MULTILABEL_PER_CLASS_AUC_WITH_SIGMOID:
        return build_multilabel_auc_metric(
            average_type='macro',
            num_classes=num_classes,
        )
    elif metric_type == MetricType.MULTICLASS_TO_BINARY_AUC:
        return build_multilabel_to_binary_auc_metric()
    elif metric_type == MetricType.MEAN_MULTILABEL_AUC:
        return build_multilabel_auc_metric(
            average_type='micro',
            num_classes=num_classes,
        )
    elif metric_type == MetricType.MEAN_MULTILABEL_PER_CLASS_AUPRC_WITH_SIGMOID:
        return build_multilabel_auprc_metric(
            average_type='macro',
            num_classes=num_classes,
        )
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)

def build_multilabel_auc_metric(average_type,num_classes):
    if num_classes == 1:
        metrics: Dict[str, Metric] = {
            f"{average_type}": BinaryAUROC(),
        }
    else:
        metrics: Dict[str, Metric] = {
            f"{average_type}": MultilabelAUROCWithSigmoid(num_labels=num_classes, average=average_type),
            f"per_class": MultilabelAUROCWithSigmoid(num_labels=num_classes,average=None)
        }
    return MetricCollection(metrics)

def build_multilabel_to_binary_auc_metric():
    metrics: Dict[str, Metric] = {
        'macro': MultilabelToBinaryAUROC(),
    }
    return MetricCollection(metrics)

def build_multilabel_auprc_metric(average_type,num_classes):
    metrics: Dict[str, Metric] = {
        f"{average_type}": MultilabelAUPRCWithSigmoid(num_labels=num_classes, average=average_type),
        f"per_class": MultilabelAUPRCWithSigmoid(num_labels=num_classes,average=None)
    }
    return MetricCollection(metrics)

def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()
