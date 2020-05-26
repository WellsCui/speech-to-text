#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d
from wave_net_layer import WaveNetLayer
from wave_net_utils import fill_voices_data_with_pads
from typing import List, Tuple, Dict, Set, Union, Optional


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

    def forward(self, input: List[List[int]], context: Optional[np.array]) -> (torch.Tensor, List[int]):
        """ Take a tensor with shape (B, N)

        @param input (torch.Tensor): a tensor with shape (B, N)
        @param context (np.array): a np.array with shape (B, N, H)
        N: number of samples
        B: batch size
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_size)
        """
        padded_input, lengths = fill_voices_data_with_pads(input)
        input_tensor = torch.tensor(
            padded_input, dtype=torch.float, device=self.device)/(65536/2)
        if context is not None:
            context = torch.from_numpy(context).to(device=self.device)
        layer_input = input_tensor.unsqueeze(1)
        layer_output = self.layers_forward(layer_input, context)
        return self.masks(layer_output, lengths), lengths

    def layers_forward(self, layer_input: torch.Tensor, context: torch.Tensor, padding=True) -> torch.Tensor:
        layer_output_aggregate = None
        for layer_index in range(self.layer_nums):
            if layer_index == 0:
                layer_output = self.layers[layer_index](layer_input, padding)
                layer_output_aggregate = layer_output
            else:
                layer_output = self.layers[layer_index](
                    layer_input, context, padding)
                layer_output_aggregate = layer_output_aggregate + layer_output
            layer_input = layer_input + layer_output
        aggregate = self.agregate1x1(F.relu(layer_output_aggregate))
        output = self.output1x1(F.relu(aggregate))
        return F.log_softmax(output, 1)

    def masks(self, output: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ return masked oupt.

        @param output (Tensor): encodings of shape (B, H, N), where B = batch size,
                                     N = max source length, H = output_size.
        @param source_lengths (List[int]): List of actual lengths for each of the voice in the batch.

        @returns masked_output (Tensor): Tensor of sentence masks of shape (B, H, N),
                                    where B = batch size, N = max source length, H = output_size.
        """
        masks = torch.zeros(output.size(0), output.size(1),
                            output.size(2), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            masks[e_id, :, :src_len] = 1
        masks = masks.to(self.device)
        return output*masks

    def reconstruct_from_output(self, source: torch.Tensor, source_lengths: List[int]) -> List[List[int]]:
        """ reconstruct voice from forward output.

        @param source (torch.Tensor): a torch.Tensor with shape (B, C, N), where B = batch size,
                                     N = max source length, C = channel_size.
        @param source_lengths (int): A list of lengths of source
        """
        voices = []
        for e_id, src_len in enumerate(source_lengths):
            # voice = source[e_id, :, :src_len].cpu().detach().numpy()
            voice_softmax = source[e_id, :, :src_len]
            voice = (torch.argmax(voice_softmax, axis=0)-128) * 256
            voices.append(voice)
        return voices

    def generate_voices(self, context: np.array) -> List[List[int]]:
        """ generate voice with context.

         @param context (np.array): a np.array with shape (B, N, H)
                                    N: number of samples
                                    B: batch size
                                    H: context size

        """
        sample_num = context.shape[1]
        batch_size = context.shape[0]
        samples = []
        context_tensor = torch.tensor(
            context, dtype=torch.float, device=self.device)
        receptive_fields = []
        for layer_index in range(self.layer_nums):
            layer = self.layers[layer_index]
            receptive_fields_channel = self.layer_channels
            if layer_index == 0:
                receptive_fields_channel = 1
                receptive_fields_size = 2
            else:
                receptive_fields_size = layer.dilation * \
                    (self.kernel_size-1) + 1
            receptive_fields.append(torch.zeros(
                batch_size, receptive_fields_channel, receptive_fields_size, device=self.device))

        for sample_index in range(sample_num):
            layer_output_aggregate = torch.zeros(
                batch_size, receptive_fields_channel, 1, device=self.device)
            ctx = context_tensor[:, sample_index:sample_index+1, :]
            for layer_index in range(self.layer_nums):
                layer_input = receptive_fields[layer_index]
                layer = self.layers[layer_index]
                if layer_index == 0:
                    layer_output = layer(layer_input, padding=False)
                else:
                    layer_output = layer(layer_input, ctx, padding=False)
                layer_output_aggregate = layer_output_aggregate + layer_output
                if layer_index < self.layer_nums-1:
                    receptive_fields[layer_index+1] = torch.cat(
                        [receptive_fields[layer_index+1][:, :, 1:], layer_output], dim=2)
            aggregate = self.agregate1x1(F.relu(layer_output_aggregate))
            output = self.output1x1(F.relu(aggregate))
            output = F.log_softmax(output, 1)
            sample = self.reconstruct_from_output(output, [1]*batch_size)
            samples_tensor = torch.stack(sample)
            receptive_fields[0] = torch.cat(
                [receptive_fields[0][:, :, 1:], samples_tensor.unsqueeze(1)/(65536/2)], dim=2)
            samples.append(samples_tensor)
        voices = torch.cat(samples, dim=1)
        return voices.numpy()

    def to(self, device: Optional[Union[int, torch.device]] = ..., dtype: Optional[Union[torch.dtype, str]] = ...,
           non_blocking: bool = ...):
        self.device = device
        self.agregate1x1.to(self.device)
        self.output1x1.to(self.device)
        for layer_index in range(self.layer_nums):
            self.layers[layer_index].to(self.device)
        return self
