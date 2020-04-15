#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

# YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size):
        """ Init Highway Model.

        @param embed_size (int): Word Embedding size 
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.projection = nn.Linear(embed_size, embed_size, bias=True)
        self.gate = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """ Take a tensor of shape (b, embed_size), compute highway model

        @param X (torch.Tensor): list of source sentence tokens

        @returns output (torch.Tensor): a variable/tensor of shape (b, embed_size)
        """
        project = torch.relu(self.projection(X))
        gate = torch.sigmoid(self.projection(X))
        output = gate * project + (1-gate) * X


        return output
# END YOUR CODE
