# coding: utf-8
"""
Implement of Forward-Forward Algorithm (FF).
Author: Mimi
Date: 2023-07-15
"""
from typing import List, Tuple

import torch
import torch.nn as nn


class FFLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cpu",
        lr: float = 0.01,
        goodness_threshold: float = 2.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
        )
        self.goodness_threshold = goodness_threshold
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.relu = torch.nn.ReLU()

    def linear_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        # L2 Norm & smoothy TODO: why???
        inputs_l2_norm = inputs.norm(2, 1, keepdim=True) + 1e-4
        
        # Normalization
        inputs /= inputs_l2_norm

        # Linear transformation
        outputs = torch.mm(inputs, self.weight.T) + self.bias.unsqueeze(0)
        return self.relu(outputs)

    def forward(
        self,
        pos_inputs: torch.Tensor,
        neg_inputs: torch.Tensor,
        train_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute `goodness`
        pos_outputs = self.linear_transform(pos_inputs)
        neg_outputs = self.linear_transform(neg_inputs)
        pos_goodness = pos_outputs.pow(2).mean(1)
        neg_goodness = neg_outputs.pow(2).mean(1)

        # Clean the layer optimizer
        self.optimizer.zero_grad()

        # Compute loss
        pos_loss = self.goodness_threshold - pos_goodness
        neg_loss = neg_goodness - self.goodness_threshold
        loss = torch.log(1 + torch.exp(torch.cat([pos_loss, neg_loss]))).mean()

        # Update the weights & bias of the layer
        if train_mode:
            loss.backward()
            self.optimizer.step()

        return pos_outputs.detach(), neg_outputs.detach(), loss.detach()
    
    
# TODO: This replemented cannot convert into DEVICE we set...
class FFClassifier(torch.nn.Module):
    def __init__(self, dims: List[int], device: str) -> None:
        super().__init__()
        self.layers = [
            FFLinear(
                in_features=dims[i],
                out_features=dims[i+1],
                lr=0.01,
                device=device,
            ) for i in range(len(dims)-1)
        ]
        
    def forward(
        self,
        pos_inputs: torch.Tensor,
        neg_inputs: torch.Tensor,
        train_mode: bool = True,
    ) -> torch.Tensor:
        total_loss = 0.0
        for layer in self.layers:
            pos_inputs, neg_inputs, loss = layer(pos_inputs, neg_inputs, train_mode)
            total_loss += loss.item()

        return total_loss

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor, num_classes: int = 10) -> int:
        goodness = 0
        for idx, layer in enumerate(self.layers):
            inputs = layer.linear_transform(inputs)
            if idx > 0:
                goodness += inputs.pow(2).mean(1)

        return torch.argmax(goodness)

