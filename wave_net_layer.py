#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d


class WaveNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, context_size=0):
        """ Init CNN Model.

        @param embed_size (int): Word Embedding size 
        """
        super(WaveNetLayer, self).__init__()
        self.dilation = dilation
        # print("WaveNetLayer.dilation:", dilation)
        self.filterConv1d = CausalConv1d(
            in_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        self.gateConv1d = CausalConv1d(
            in_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        self.adjustConv1d = nn.Conv1d(out_channels, in_channels, 1)
        self.context_size = context_size
        if context_size > 0:
            self.context_filter_projection = nn.Linear(
                context_size, out_channels)
            self.context_gate_projection = nn.Linear(
                context_size, out_channels)
        
    
    def forward(self, X: torch.Tensor, context: torch.Tensor, padding=True) -> torch.Tensor:
        """ Take a tensor with shape (B, C, N)

        @param X (torch.Tensor): a tensor with shape (B, C, N)
        @param Y (torch.Tensor): a tensor with shape (B, N, H)
        N: number of samples
        B: batch size
        C: input channel
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_channels)
        """
        filtered = torch.tanh(self.filterConv1d(X, padding))
        gated = torch.sigmoid(self.gateConv1d(X, padding))
        
        if self.context_size > 0:
            filtered_with_context = filtered + \
                self.context_filter_projection(context).transpose(1, 2)
            gated_with_context = gated + \
                self.context_gate_projection(context).transpose(1, 2)
            output = filtered_with_context * gated_with_context
        else:
            output = filtered * gated
        output = self.adjustConv1d(output)
        return output
    
