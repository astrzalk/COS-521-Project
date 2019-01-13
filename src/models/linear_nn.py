#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from matrix_helpers import balanced_init, independent_init


class LinearNet(nn.Module):
    def __init__(self, layers, hidden_width, input_size, output_size, bias=False, p=0.0):
        super(LinearNet, self).__init__()
        self.num_layers = layers
        self.width = hidden_width
        self.output_size = output_size
        self.input_size = input_size
        self.dropout = nn.Dropout(p)

        self.layer_sizes = [input_size] + [self.width] * (self.num_layers - 1) + \
                           [output_size]
        layers = [[nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1], bias=bias), self.dropout]
                   for i in range(self.num_layers)]
        # [:-1] : Quick hack to ensure dropout is only in between layers not at the end
        self.layers = nn.ModuleList([item for sublist in layers for item in sublist][:-1])

    # forward: do the computation for a linear net
    def forward(self, x):
        out = x
        for i,l in enumerate(self.layers):
            out = l(out)
        return out.t()

    # init_balanced: initialize weights according to
    # "Balanced Initialization" scheme
    def init_balanced(self, W, randomize=True):
        W_arr = balanced_init(W, self.num_layers, self.width, randomize=randomize)

        i = 0 # Internal counter to avoid dropout layers
        for _,l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                l.weight.data = nn.Parameter(torch.tensor(W_arr[i]).to(torch.float))
                i+=1

    # init_unbalanced: initialize weights independently
    def init_independent(self, scale):
        W_arr, b_arr = independent_init(self.layer_sizes, scale)

        i = 0 # Internal counter to avoid dropout layers
        for _,l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                l.weight.data = nn.Parameter(torch.tensor(W_arr[i]).to(torch.float))
                i+=1

        return b_arr

    # get_wts: return the weight matrices of the linear network
    def get_wts(self):
        wts = []
        for _,l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                wts.append(l.weight.data)

        return wts

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

def get_delta(weight_mats):
    """
    :weight_mats: A list of weight matrices for each layer in the neural net.

    :returns: delta: the balancedness value, i.e. minimum delta such that
    \norm{W_{j+1}^TW_{j+1} - W_jW_j^T}_F \leq \delta
    """
    deltas = []
    for ind, _ in enumerate(weight_mats):
        if ind == 0:
            continue
        W_first = weight_mats[ind - 1]
        W_second = weight_mats[ind]

        diff = W_second.transpose(0, 1) @ W_second -\
               W_first @ W_first.transpose(0, 1)
        balance_val = torch.norm(diff)
        deltas.append(balance_val)
    return deltas

def get_layer_norms(weight_mats):
    layer_norms = []
    for W in weight_mats:
        layer_norm = torch.norm(W @ W.transpose(0,1))
        layer_norms.append(layer_norm)
    return layer_norms

def train(model, loss_fn, X, y, learning_rate, eps=1e-5, verbose=False):
    """
    model         : Type = torch.nn.Module; the neural net model already initialized
    loss_fn       : Type = torch.nn.Module (technically); just the loss function
    X             : Type = pytorch tensor; Dataset
    y             : Type = pytorch tensor; labels
    learning_rate : Type = float; learning rate for gradient descent
    eps           : Type = float; tolerance for how close loss is to global_opt
    global_opt    : Type = float; the global optimum for this problem
    verbose       : Type = Boolean; If true prints out each iteration, loss pair
    """
    loss = np.inf
    num_iter = 0
    deltas, l_norms = [], []
    while loss > eps:
        # Get Balancedness value by iterating through the weights
        weight_mats = model.get_wts()

        # weight_mats = [layer.weight.data for layer in model.children() if isinstance(layer, nn.Linear)]
        delta = get_delta(weight_mats)
        deltas.append(delta)
        l_norm = get_layer_norms(weight_mats)
        l_norms.append(l_norm)

        W = model(torch.eye(128)) # W_N * W_{N - 1} * ... * W_1

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        loss = loss_fn(W, X, y)
        if verbose:
            print(num_iter, loss.item())

        model.zero_grad()
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                param.data -= learning_rate * param.grad

        num_iter += 1
        if num_iter > 1e6: # Breaks training if it takes too long to converge
            break
        if num_iter > 1e5:
            print(num_iter, loss)
    return num_iter, loss, deltas, l_norms
