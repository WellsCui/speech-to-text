#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VoiceCNN(nn.Module):
    """ Simple CNN Model:
        - 1D CNN
    """

    def __init__(self, output_channels, kernel_size):
        """ Init CNN Model.

        @param embed_size (int): Word Embedding size 
        """
        super(VoiceCNN, self).__init__()
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(1, output_channels, kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Take a tensor with shape (B, N, S)

        @param X (torch.Tensor): a tensor with shape (B, N, S)
        N: number of syllables
        B: batch size
        S: syllable size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_channels)
        """
        tensors = []
        for batch_index in range(input.shape[0]):
            x_conv = self.conv1d(input[batch_index].unsqueeze(1))
            x_max, _ = torch.relu(x_conv).max(dim=-1, keepdim=False)
            tensors.append(x_max) 
        output = torch.stack(tensors) 
        return output

### END YOUR CODE

