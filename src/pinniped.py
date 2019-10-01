#!/usr/bin/python3

import sys
import argparse
import scipy.io.arff
import torch
import math
from collections import OrderedDict

"""
Command line arguments
"""
parser = argparse.ArgumentParser(description='Parameterized Interactive Neural Network Iteratively Plotting Experimental Decisions')
parser.add_argument('--arff', type=str, required=True)
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--test', action='store_true')
mode.add_argument('--train', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--learning-rate', type=float, default=0.5)
parser.add_argument('--momentum', type=float) # @TODO: Implement me!
parser.add_argument('--activation-unit', type=str, choices=['sigmoid', 'tanh', 'ReLU'], default='sigmoid')
parser.add_argument('--layer-dims', type=str, required=True)

activation_units = { 'sigmoid' : torch.nn.Sigmoid, 'tanh' : torch.nn.Tanh, 'ReLU' : torch.nn.ReLU }

"""
Encode a series of amplitudes in the range [0,1] according to the series dynamic range.
"""
def compress(x):
    dmax = max(x)
    dmin = min(x)
    return tuple([(d - dmin)/(dmax - dmin) for d in x])

"""
Load training data from ARFF input file.
"""
def load_arff(arff_fname):
    arff_data, arff_meta = scipy.io.arff.loadarff(arff_fname)

    # Convert ARFF to torch tensors.
    X, Y = zip(*[ (XY[0:-1], XY[-1].decode('UTF-8')) for XY in [ tuple(nd) for nd in arff_data ] ])
    L = OrderedDict()
    classes = len(arff_meta['target'][1])
    for c in range(classes):
        y = arff_meta['target'][1][c]
        L[y] = torch.zeros(classes)
        L[y][c] = 1.0
    X = [torch.tensor(compress(x)) for x in X]
    Y = [L[y] for y in Y]
    return X, Y, L

def d_Theta(W_i, W_j):
    return [(torch.norm(w_j.sub(w_i)).item(), d_theta(w_i, w_j)) for (w_i, w_j) in zip(W_i, W_j)]

def d_theta(w_i, w_j):
    norm_w_ij = torch.norm(w_i) * torch.norm(w_j)
    if norm_w_ij.item() == 0.0:
        # Any zero-length vector can be considered to be simultaneously colinear and not with any other.
        # Erring on the side of colinearity and avoiding div by zero.
        return 0.0
    else:
        # Clamping per https://github.com/pytorch/pytorch/issues/8069
        return torch.acos(torch.clamp(torch.dot(w_i, w_j) / norm_w_ij, min=-1.0, max=1.0)).item()

"""
Train NN.
"""
def train_nn(model, X, Y):
    one_hots = torch.nn.functional.one_hot(torch.arange(0,model[-1].out_features)).type(torch.float)
    for epoch in range(args.epochs):
        passed = 0
        failed = 0
        LW_i = [ layer.weight.data.clone().detach().requires_grad_(True) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        for x, y in zip(X,Y):
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            y_label = one_hots[y_pred.argmax().item()]
            if torch.eq(y, y_label).all() :
                passed = passed + 1
            else:
                failed = failed + 1

            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    p -= args.learning_rate * p.grad
                model.zero_grad()
        LW_j = [ layer.weight.data.clone().detach().requires_grad_(True) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        dTheta = [d_Theta(W_i, W_j) for (W_i, W_j) in zip(LW_i, LW_j)]
        # @TODO: may want to add norm of difference vector for clarity, esp. with 0-length vectors.
        print("TRAIN/{}: passed = {}, failed = {}, d(w,θ) = {}".format(epoch, passed, failed, dTheta))

"""
Test NN.
"""
def test_nn(model, X, Y):
    one_hots = torch.nn.functional.one_hot(torch.arange(0,model[-1].out_features)).type(torch.float)
    for x,y in zip(X,Y):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        y_label = one_hots[y_pred.argmax.item()]
        if torch.eq(y, y_label).all() :
            passed = passed + 1
        else:
            failed = failed + 1
    print("TEST/{}: passed = {}, failed = {}".format(epoch, passed, failed))



"""
Main
"""

args = parser.parse_args()

# Get layer dims.
layer_D = [ int(D) for D in args.layer_dims.split(",") ]

if len(layer_D) < 2:
    raise argparse.ArgumentError("Must define at least two layer dimensions (input + output)")

# Determine activation unit.
activation_unit = activation_units[args.activation_unit]

# Set loss function.
# @TODO: Verify that this is the correct loss function to use.
loss_fn = torch.nn.MSELoss(reduction='sum')

# Load input (in ARFF format).
X, Y, L = load_arff(args.arff)

# Build set of layers in order.
layers = OrderedDict()
for i in range(len(layer_D)-1):
    layers['L{}'.format(i)] = torch.nn.Linear(layer_D[i], layer_D[i+1])
    if i < len(layer_D) - 2:
        layers['A{}'.format(i)] = activation_unit()

# Create NN model.
model = torch.nn.Sequential(layers)
print(model, file=sys.stderr)

# Train or test, depending on user spec.
# @TODO: Save trained model for future loading. Otherwise, have to train before testing.
if args.train:
    train_nn(model, X, Y)
elif args.test:
    test_nn(model, X, Y)
