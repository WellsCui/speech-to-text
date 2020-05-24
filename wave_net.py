#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d
from wave_net_layer import WaveNetLayer
from wave_net_utils import fill_voices_data_with_pads
from typing import List, Tuple, Dict, Set, Union
class WaveNet(nn.Module):
    """ Simple CNN Model:
        - 1D CNN
    """

    def __init__(self, layer_nums, kernel_size=2, layer_channels=16, target_size=256, context_size=0):
        """ Init CNN Model.

        @param layer_nums (int): Number of layers
        @param context_size (int): Size of conditional context
        """
        super(WaveNet, self).__init__()

        self.layer_nums = layer_nums
        self.layer_channels = layer_channels
        self.context_size = context_size
        self.kernel_size = kernel_size

        self.layers = []
        self.agregate_size = 64
        self.output_size = target_size

        self.agregate1x1 = nn.Conv1d(
            self.layer_channels, self.agregate_size, 1)
        self.output1x1 = nn.Conv1d(self.agregate_size, self.output_size, 1)

        for layer_index in range(self.layer_nums):
            # print("building layer:", layer_index)
            dilation = 2 ** (layer_index % 10)
            if layer_index == 0:
                self.layers.append(CausalConv1d(
                    1, self.layer_channels, self.kernel_size))
            else:
                self.layers.append(WaveNetLayer(
                    self.layer_channels, self.layer_channels, self.kernel_size, dilation, context_size=self.context_size))

    def forward(self, input: List[List[int]], context: torch.Tensor) -> torch.Tensor:
        """ Take a tensor with shape (B, N)

        @param input (torch.Tensor): a tensor with shape (B, N)
        @param context (torch.Tensor): a tensor with shape (B, N, H)
        N: number of samples
        B: batch size
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_size)
        """
        padded_input, lengths = fill_voices_data_with_pads(input)
        input_tensor = torch.tensor(padded_input, dtype=torch.float, device=self.device)/(65536/2)
        layer_input = input_tensor.unsqueeze(1)
        layer_output_aggregate = None
        for layer_index in range(self.layer_nums):
            # print("running layer:", layer_index)
            if layer_index == 0:
                layer_output = self.layers[layer_index](layer_input)
                layer_output_aggregate = layer_output
            else:
                layer_output = self.layers[layer_index](layer_input, context)
                layer_output_aggregate = layer_output_aggregate + layer_output
            layer_input = layer_input + layer_output
        aggregate = self.agregate1x1(F.relu(layer_output_aggregate))
        output = self.output1x1(F.relu(aggregate))
        output = F.log_softmax(output, 1)
        return self.masks(output, lengths)

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.agregate1x1.weight.device

    
    def masks(self, output: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ return masked oupt.

        @param output (Tensor): encodings of shape (B, H, N), where B = batch size,
                                     N = max source length, H = output_size.
        @param source_lengths (List[int]): List of actual lengths for each of the voice in the batch.

        @returns masked_output (Tensor): Tensor of sentence masks of shape (B, H, N),
                                    where B = batch size, N = max source length, H = output_size.
        """
        masks = torch.zeros(output.size(0), output.size(1), output.size(2), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            masks[e_id, :, :src_len] = 1
        masks = masks.to(self.device)
        return output*masks
