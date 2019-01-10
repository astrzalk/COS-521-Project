#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class three_layer_nn(nn.Module):
    """
    A three layer linear nn with input, hidden and output dims
    as specified in the paper 'A Convergence Analysis of Gradient Descent
    for Deep Linear Neural Networks'.
    Instead of outputing
    """

    def __init__(self, init_type='normal', std=1, do_dropout=False, p=0.5):
        super(three_layer_nn, self).__init__()
        self.mu = 0
        self.s = std
        self.do_drop = do_dropout
        if init_type == 'normal':
            self.layer_1 = nn.Linear(128, 32)
            self.layer_2 = nn.Linear(32, 32)
            self.layer_3 = nn.Linear(32, 1)
            nn.init.normal_(self.layer_1.weight, self.mu, self.s)
            nn.init.normal_(self.layer_2.weight, self.mu, self.s)
            nn.init.normal_(self.layer_3.weight, self.mu, self.s)
        elif init_type == 'balanced':
            # Sample A as a D_out by D_in Normal matrix
            A = torch.empty(1, 128)
            nn.init.normal_(A, self.mu, self.s)

            # Do SVD by hand for vector is easy
            # Check out:
            #https://math.stackexchange.com/questions/1181800/
            #singular-value-decomposition-of-column-row-vectors
            # U and V are swapped, the below is Economically SVD
            # Padded with zeros as in the paper
            U = torch.zeros(1, 32)
            U[0, 0] = 1.0 # be careful for type

            S = torch.zeros(32, 32)
            s_1 = torch.norm(A).pow(1 / 3)
            S[0, 0] = s_1

            V = torch.zeros(128, 32)
            V[:, 0] = A / torch.norm(A)

            self.layer_1 = nn.Linear(128, 32)
            self.layer_2 = nn.Linear(32, 32)
            self.layer_3 = nn.Linear(32, 1)
            # Layer 1: S^{1/3} * V.T; 32 by 128
            self.layer_1.weight.data = S @ V.transpose(0,1)
            # Layer 2: S^{1/3}; 32 by 32
            self.layer_2.weight.data = S
            # Layer 3: U * S^{1/3}; 1 by 32
            self.layer_3.weight.data = U @ S

        if self.do_drop:
            self.p = p
            self.drop = nn.Dropout(self.p)

    def forward(self):
        W_1 = self.layer_1.weight
        W_2 = self.layer_2.weight
        W_3 = self.layer_3.weight
        if self.do_drop:
            W = self.drop(W_3) @ self.drop(W_2) @ self.drop(W_1) # only works in python 3.5+, @ = mat mul
        else:
            W = W_3 @ W_2 @ W_1 # only works in python 3.5+, @ = mat mul

        return W

class eight_layer_nn(nn.Module):
    """
    An eight layer linear nn with input, hidden and output dims
    as specified in the paper 'A Convergence Analysis of Gradient Descent
    for Deep Linear Neural Networks'.
    Instead of outputing
    """

    def __init__(self, init_type='normal', std=1, do_dropout=False, p=0.5):
        super(eight_layer_nn, self).__init__()
        self.s = std
        self.do_drop = do_dropout
        if init_type == 'normal':
            self.layer_1 = nn.Linear(128, 32)
            self.layer_2 = nn.Linear(32, 32)
            self.layer_3 = nn.Linear(32, 32)
            self.layer_4 = nn.Linear(32, 32)
            self.layer_5 = nn.Linear(32, 32)
            self.layer_6 = nn.Linear(32, 32)
            self.layer_7 = nn.Linear(32, 32)
            self.layer_8 = nn.Linear(32, 1)
            torch.nn.init.normal_(self.layer_1.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_2.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_3.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_4.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_5.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_6.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_7.weight, 0, self.s)
            torch.nn.init.normal_(self.layer_8.weight, 0, self.s)
        elif init_type == 'balanced':
            # Sample A as a D_out by D_in Normal matrix
            A = torch.empty(1, 128)
            nn.init.normal_(A, self.mu, self.s)

            # Same as in three layer_nn
            U = torch.zeros(1, 32)
            U[0, 0] = 1.0 # be careful for type

            S = torch.zeros(32, 32)
            s_1 = torch.norm(A).pow(1 / 8)
            S[0, 0] = s_1

            V = torch.zeros(128, 32)
            V[:, 0] = A / torch.norm(A)

            self.layer_1 = nn.Linear(128, 32)
            self.layer_2 = nn.Linear(32, 32)
            self.layer_3 = nn.Linear(32, 1)
            # Layer 1: S^{1/3} * V.T; 32 by 128
            self.layer_1.weight.data = S @ V.transpose(0,1)
            # Layers 2-7: S^{1/3}; 32 by 32
            self.layer_2.weight.data = S
            self.layer_3.weight.data = S
            self.layer_4.weight.data = S
            self.layer_5.weight.data = S
            self.layer_6.weight.data = S
            self.layer_7.weight.data = S
            # Layer 3: U * S^{1/3}; 1 by 32
            self.layer_8.weight.data = U @ S

        if self.do_drop:
            self.p = p
            self.drop = nn.Dropout(self.p)

    def forward(self):
        if self.do_drop:
            W_1, W_2 = self.drop(self.layer_1.weight), self.drop(self.layer_2.weight)
            W_3, W_4 = self.drop(self.layer_3.weight), self.drop(self.layer_4.weight)
            W_5, W_6 = self.drop(self.layer_5.weight), self.drop(self.layer_6.weight)
            W_7, W_8 = self.drop(self.layer_7.weight), self.drop(self.layer_8.weight)
            W = W_8 @ W_7 @ W_6 @ W_5 @ W_4 @ W_3 @ W_2 @ W_1 # only works in python 3.5+, @ = mat mul
        else:
            W_1, W_2 = self.layer_1.weight, self.layer_2.weight
            W_3, W_4 = self.layer_3.weight, self.layer_4.weight
            W_5, W_6 = self.layer_5.weight, self.layer_6.weight
            W_7, W_8 = self.layer_7.weight, self.layer_8.weight
            W = W_8 @ W_7 @ W_6 @ W_5 @ W_4 @ W_3 @ W_2 @ W_1
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

def train(model, loss_fn, X, y, learning_rate, global_opt, eps=1e-11, verbose=False):
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
    while torch.abs(global_opt - loss) > eps: # I THINK THIS IS THE BUG
        W = model() # W_N * W_{N - 1} * ... * W_1

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        loss = loss_fn(W, X, y)
        if verbose:
            print(num_iter, loss.item())

        # Zero the gradients before running the backward pass.
        # In pytorch, gradients are accumulated with .backward(), hence,
        # we need to zero them out each round
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
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
    return num_iter, loss

