# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional
from soft_moe_pytorch import SoftMoE
from torch import Tensor, nn
import torch

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MaxZeroTanhshrink(nn.Module):
    def __init__(self):
        super(MaxZeroTanhshrink, self).__init__()
        self.tanhshrink = nn.Tanhshrink()
    def forward(self, x):
        return torch.clamp(self.tanhshrink(x), min=0)

class MixtureActivationMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.softmax = nn.Softmax(dim=0)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        
        ###
        self.activations = [
            nn.GELU(),
            nn.SiLU(),
            nn.Softplus(),
            MaxZeroTanhshrink(),
        ]
        ###

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        # self.alpha = nn.Parameter(torch.ones(len(self.activations)))
        # self.alpha = nn.Parameter(torch.tensor([1, 0.25, 0.25, 0.25]))
        self.alpha = nn.Parameter(torch.randn(len(self.activations)))
        # self.layer_norm = nn.LayerNorm(hidden_features)


    def forward(self, x: Tensor) -> Tensor:

        alpha_norm = self.softmax(self.alpha)
        x = self.fc1(x)
        x = sum(alpha_norm[i] * self.activations[i](x) for i in range(len(self.activations)))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MixtureExpertsMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        sequence_length=None,
        drop: float = 0.0,
        num_experts: int = 4,
        num_slots=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.moe = SoftMoE(
            dim = in_features,         
            seq_len = sequence_length,
            num_experts = num_experts,
            dropout = drop,
            num_slots=num_slots,
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.moe(x)