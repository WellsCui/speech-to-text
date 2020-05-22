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
from highway import Highway

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
        self.conv1d1 = nn.Conv1d(1, output_channels//4, kernel_size)
        self.conv1d2 = nn.Conv1d(output_channels//4, output_channels//2, kernel_size)
        self.conv1d3 = nn.Conv1d(output_channels//2, output_channels, kernel_size)
        self.highway = Highway(output_channels)
        # self.dropout = nn.Dropout(p=0.2)

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
            x_conv1 = self.conv1d1(input[batch_index].unsqueeze(1))
            x_conv2 = self.conv1d2(x_conv1)
            x_conv3 = self.conv1d3(x_conv2)
            x_max, _ = torch.relu(x_conv3).max(dim=-1, keepdim=False)
            x_highway = self.highway(x_max)
            # x_output = self.dropout(x_highway)
            tensors.append(x_highway) 
        output = torch.stack(tensors) 
        return output

### END YOUR CODE

