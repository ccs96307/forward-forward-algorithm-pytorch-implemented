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

        self.classify_layer = torch.nn.Linear(dims[-1], 10).to(device)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.classify_layer.parameters(), lr=0.01)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(
        self,
        pos_inputs: torch.Tensor,
        neg_inputs: torch.Tensor,
        pos_labels: torch.Tensor,
        train_mode: bool = True,
    ) -> torch.Tensor:
        total_loss = 0.0

        # Forward layers
        for layer in self.layers:
            pos_inputs, neg_inputs, loss = layer(pos_inputs, neg_inputs, train_mode)
            total_loss += loss.item()

        # Classifier Layer (the last layer)
        pos_outputs = self.classify_layer(pos_inputs)
        pos_outputs = self.softmax(pos_outputs)
        loss = self.criterion(pos_outputs, pos_labels)
        loss.backward()
        self.optimizer.step()

        return total_loss + loss.item()

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor, num_classes: int = 10) -> int:
        for layer in self.layers:
            inputs = layer.linear_transform(inputs)

        outputs = self.classify_layer(inputs)
        outptus = self.softmax(outputs)

        return torch.argmax(outputs, dim=1)

