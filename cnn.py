#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """ Simple CNN Model:
        - 1D CNN
    """

    def __init__(self, embed_char_size, embed_word_size, kernel_size):
        """ Init CNN Model.

        @param embed_size (int): Word Embedding size 
        """
        super(CNN, self).__init__()
        self.embed_char_size = embed_char_size
        self.embed_word_size = embed_word_size
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(embed_char_size, embed_word_size, kernel_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Take a tensor of shape (b, embed_char_size), compute highway model

        @param X (torch.Tensor): list of source sentence tokens

        @returns output (torch.Tensor): a variable/tensor of shape (b, embed_size)
        """
        x_conv = self.conv1d(X)
        x_conv_out = F.max_pool1d(torch.relu(x_conv), X.shape[2] - self.kernel_size + 1)
        return torch.squeeze(x_conv_out, 2)

### END YOUR CODE

