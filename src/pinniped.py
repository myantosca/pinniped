#!/usr/bin/python3

import sys
import argparse
import scipy.io.arff
import torch
import math
from collections import OrderedDict
import pickle

"""
Command line arguments
"""
parser = argparse.ArgumentParser(description='Parameterized Interactive Neural Network Iteratively Plotting Experimental Decisions')
parser.add_argument('--arff', type=str, required=True)
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--test', action='store_true')
mode.add_argument('--train', action='store_true')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--learning-rate', type=float, default=0.5)
parser.add_argument('--momentum', type=float, default = 0.0)
parser.add_argument('--activation-unit', type=str.lower, choices=['sigmoid', 'tanh', 'relu'], default='sigmoid')
parser.add_argument('--layer-dims', type=str, required=True)
reserved = parser.add_mutually_exclusive_group()
reserved.add_argument('--reserve-frac', type=float, default=None)
reserved.add_argument('--reserve-every', type=int, default=None)
parser.add_argument('--sgd', action='store_true')
activation_units = { 'sigmoid' : torch.nn.Sigmoid, 'tanh' : torch.nn.Tanh, 'relu' : torch.nn.ReLU }

"""
Encode a series of amplitudes in the range [0,1] according to the series dynamic range.
"""
def compress(x):
    dmax = max(x)
    dmin = min(x)
    return torch.tensor([(d - dmin)/(dmax - dmin) for d in x]).type(torch.double)

"""
Load training data from ARFF input file.
"""
def load_arff(arff_fname):
    arff_data, arff_meta = scipy.io.arff.loadarff(arff_fname)

    # Convert ARFF to torch tensors.
    X, Y = zip(*[ (XY[0:-1], XY[-1].decode('UTF-8')) for XY in [ tuple(nd) for nd in arff_data ] ])
    L = OrderedDict()
    classes = len(arff_meta['target'][1])
    one_hots = torch.nn.functional.one_hot(torch.arange(0,classes)).type(torch.double)
    for c in range(classes):
        y = arff_meta['target'][1][c]
        L[y] = one_hots[c]
    X = torch.stack([compress(x) for x in X ])
    Y = torch.stack([ L[y] for y in Y ])
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
    classes = model[-1].out_features
    sample_count = X.size()[0]
    confusion_matrix = torch.zeros(classes, classes).type(torch.int)
    one_hots = torch.nn.functional.one_hot(torch.arange(0,classes)).type(torch.double)
    sample_indices = torch.arange(0,sample_count)
    grad_bias = [p.data.new_zeros(p.size()) for p in model.parameters()]

    for layer in [layer for layer in model.children() if type (layer) if type(layer) is torch.nn.Linear]:
        layer.weight.data.copy_(torch.randn_like(layer.weight.data))

    if args.sgd:
        # Stochastic gradient descent: shuffle
        sample_indices = torch.randperm(sample_count)

    if args.reserve_frac is not None:
        # Reserve the last fraction
        reserved = math.ceil(sample_count * args.reserve_frac)
        training_indices = sample_indices[0:-reserved]
        reserved_indices = sample_indices[-reserved:]
    elif args.reserve_every is not None:
        # Reserve at regular intervals
        training_indices = torch.stack([i for i in sample_indices if i % args.reserve_every != 0])
        reserved_indices = sample_indices[0::args.reserve_every]


    for epoch in range(args.epochs):
        if args.sgd:
            # Stochastic gradient descent: shuffle
            training_indices = torch.stack([training_indices[i] for i in torch.randperm(len(training_indices))])
        passed = 0
        failed = 0
        LW_i = [ layer.weight.data.clone().detach().requires_grad_(True) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        # Permute the order of the data for stochastic batch descent.
        reserved_X = X.index_select(0, reserved_indices)
        reserved_Y = Y.index_select(0, reserved_indices)

        batch = 0
        offset = 0
        while offset < len(training_indices):
            batch_indices = training_indices[offset:offset+args.batch_size]
            offset += args.batch_size
            batch_X = X.index_select(0, batch_indices)
            batch_Y = Y.index_select(0, batch_indices)

            # Train the model on the input X
            trained_Y = model(batch_X)
            # Make label predictions from argmax of the output.
            trained_labels = torch.stack([one_hots[y.argmax().item()] for y in trained_Y ])
            trained_hits = batch_Y.eq(trained_labels).all(1).to(torch.int).sum()
            trained_misses = len(batch_Y) - trained_hits

            # Calculate losses for back-propagation.
            loss = loss_fn(trained_Y, batch_Y)
            loss.backward()

            # Turn off autograd so as to not pollute the gradients with validation set nor the backpropagation itself.
            with torch.no_grad():
                b_p = 0
                for p in model.parameters():
                    p -= args.learning_rate * ((1 - args.momentum) * p.grad + args.momentum * grad_bias[b_p])
                    grad_bias[b_p].copy_(p.grad)
                    b_p += 1
                model.zero_grad()
                validated_Y = model(reserved_X)
                validated_labels = torch.stack([one_hots[y.argmax().item()] for y in validated_Y])
                validated_hits = reserved_Y.eq(validated_labels).all(1).to(torch.int).sum()
                validated_misses = len(reserved_Y) - validated_hits

            print("TRAIN[{},{}]: {}/{}".format(epoch, batch, trained_hits, len(batch_indices)))
            for i in range(len(batch_indices)):
                confusion_matrix[batch_Y[i].argmax().item()][trained_Y[i].argmax().item()] +=1
            print_matrix(confusion_matrix)
            confusion_matrix.fill_(0)

            print("VALID[{},{}]: {}/{}".format(epoch, batch, validated_hits, len(reserved_indices)))
            for i in range(len(reserved_indices)):
                confusion_matrix[reserved_Y[i].argmax().item()][validated_Y[i].argmax().item()] +=1
            print_matrix(confusion_matrix)
            confusion_matrix.fill_(0)

            batch += 1

        LW_j = [ layer.weight.data.clone().detach().requires_grad_(True) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        dTheta = [d_Theta(W_i, W_j) for (W_i, W_j) in zip(LW_i, LW_j)]
        #print("d(w,θ) = {}".format(dTheta))


def print_matrix(M):
    s = ""
    for r in range(M.size()[0]):
        s += "|"
        for c in range(M.size()[1]):
          s += "{}{}|".format('*' if r == c else ' ', M[r][c])
        s += "\n"
    print(s)
"""
Test NN.
"""
def test_nn(model, X, Y):
    classes = model[-1].out_features
    confusion_matrix = torch.zeros(classes, classes)
    one_hots = torch.nn.functional.one_hot(torch.arange(0,classes)).type(torch.double)
    passed = 0
    failed = 0
    with torch.no_grad():
        predicted_Y = model(X)
        predicted_labels = torch.stack([one_hots[y.argmax().item()] for y in predicted_Y])
        predicted_hits = Y.eq(predicted_labels).all(1).to(torch.int).sum()
        for i in range(Y.size()[0]):
            confusion_matrix[Y[i].argmax().item()][predicted_Y[i].argmax().item()] +=1

    print("TEST: {}/{}".format(predicted_hits, Y.size()[0]))
    print(confusion_matrix)



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
    layers['L{}'.format(i)] = torch.nn.Linear(layer_D[i], layer_D[i+1]).to(torch.double)
    if i < len(layer_D) - 2:
        layers['A{}'.format(i)] = activation_unit().to(torch.double)

model = torch.nn.Sequential(layers)
# Create NN model.
print(model, file=sys.stderr)

# Train or test, depending on user spec.
# @TODO: Save trained model for future loading. Otherwise, have to train before testing.
if args.train:
    train_nn(model, X, Y)
    with open('./model.nn', 'wb') as fout:
        pickle.dump(model.state_dict(), fout)
elif args.test:
    with open('./model.nn', 'rb') as fin:
        model.load_state_dict(pickle.load(fin))
    test_nn(model, X, Y)
