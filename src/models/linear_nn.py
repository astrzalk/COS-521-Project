#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class three_layer_nn(nn.Module):
    """
    A three layer linear nn with input, hidden and output dims
    as specified in the paper 'A Convergence Analysis of Gradient Descent
    for Deep Linear Neural Networks'.
    Instead of outputing
    """

    def __init__(self, init_type='normal', std=1, do_dropout=False, p=0.5, seed=521):
        super(three_layer_nn, self).__init__()
        torch.manual_seed(seed)
        self.mu = 0
        self.s = std
        self.do_dropout = do_dropout
        if init_type == 'normal':
            self.layer_1 = nn.Linear(128, 32)
            self.layer_2 = nn.Linear(32, 32)
            self.layer_3 = nn.Linear(32, 1)
            torch.nn.init.normal_(self.layer_1.weight, self.mu, self.s)
            torch.nn.init.normal_(self.layer_2.weight, self.mu, self.s)
            torch.nn.init.normal_(self.layer_3.weight, self.mu, self.s)
        elif init_type == 'balanced':
            raise NotImplementedError
        if self.do_dropout:
            self.p = p
            self.drop = nn.Dropout(self.p)

    def forward(self):
        W_1 = self.layer_1.weight
        W_2 = self.layer_2.weight
        W_3 = self.layer_3.weight
        if self.do_dropout:
            W = self.drop(W_3) @ self.drop(W_2) @ self.drop(W_1) # only works in python 3.5+, @ = mat mul
        else:
            W = W_3 @ W_2 @ W_1 # only works in python 3.5+, @ = mat mul

        return W

class eight_layer_nn(nn.Module):
    """
    A three layer linear nn with input, hidden and output dims
    as specified in the paper 'A Convergence Analysis of Gradient Descent
    for Deep Linear Neural Networks'.
    Instead of outputing
    """

    def __init__(self):
        super(eight_layer_nn, self).__init__()
        self.layer_1 = nn.Linear(128, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_3 = nn.Linear(32, 1)

    def forward(self):
        W_1 = self.layer_1.weight
        W_2 = self.layer_2.weight
        W_3 = self.layer_3.weight
        W = W_3 @ W_2 @ W_1 # only works in python 3.5+, @ = mat mul
        return W

class fro_loss(nn.Module):

    def __init__(self):
        super(fro_loss, self).__init__()

    def forward(self, W, X, y):
        """
        W: W_N * W_{N - 1} * ... * W_1 and
        X: Data, type pytorch tensor
        y: labels, type pytorch tensor
        """
        cross_cov = (1 / len(y)) * y.transpose(0,1) @ X
        loss = 0.5 * torch.norm(W - cross_cov).pow(2) # Since output_dim is 1 fro norm becomes l2-norm
        return loss

# # Taken from https://github.com/kevinzakka/pytorch-goodies
# def max_norm(model, max_val=3, eps=1e-8):
#     for name, param in model.named_parameters():
#         if 'bias' not in name:
#             norm = param.norm(2, dim=0, keepdim=True)
#             desired = torch.clamp(norm, 0, max_val)
#             param = param * (desired / (eps + norm))

# Their paper used a 3 layer neural network (2 hidden layers) each
# with dimension 32 each
# D_in = 128
# H = 32
# D_out = 1

# model = nn.Sequential(
#         # nn.Dropout(p=0.2), # p = probability of dropping
#         nn.Linear(D_in, H), # Layer 1 Weights
#         # nn.Dropout(p=0.2), # p = probability of dropping
#         nn.Linear(H, H), # Layer 2 Weights
#         # nn.Dropout(p=0.2), # p = probability of dropping
#         nn.Linear(H, D_out)  # Layer 3 Weights
#       )





