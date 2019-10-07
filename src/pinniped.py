#!/usr/bin/python3

import sys
import argparse
import scipy.io.arff
import torch
import math
from collections import OrderedDict
import pickle
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as mplp
import matplotlib.colors as mplc
import functools

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

parser.add_argument('--autograd-backprop', action='store_true')
parser.add_argument('--sgd', action='store_true')
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--activation-bins', type=int, default=20)
parser.add_argument('--plot-every', type=int, default=1)

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

def weight_change(W_i, W_j, dWNorm, dTheta):
    for layer in range(len(W_j)):
        dwnorm = torch.zeros(W_j[layer].size()[0], 1)
        dtheta = torch.zeros(W_j[layer].size()[0], 1)
        for node in range(W_j[layer].size()[0]):
            dwnorm[node] = torch.norm(W_j[layer][node].sub(W_i[layer][node])).item()
            dtheta[node] = d_theta(W_i[layer][node],W_j[layer][node])
        dWNorm[layer] = torch.cat((dWNorm[layer], dwnorm), 1)
        dTheta[layer] = torch.cat((dTheta[layer], dtheta), 1)


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
    trained_confusion = torch.zeros(classes, classes).type(torch.int)
    validated_confusion = torch.zeros(classes, classes).type(torch.int)
    one_hots = torch.nn.functional.one_hot(torch.arange(0,classes)).type(torch.double)
    sample_indices = torch.arange(0,sample_count)
    grad_bias = [p.data.new_zeros(p.size()).requires_grad_(False).type(torch.double) for p in model.parameters()]

    if args.autograd_backprop:
        for layer in [layer for layer in model.children() if type (layer) if type(layer) is torch.nn.Linear]:
            if args.debug:
                print(layer.weight.data, file=sys.stderr)
    else:
        for layer in [layer for layer in model.children() if type (layer) if type(layer) is torch.nn.Linear]:
            layer.weight.data.copy_(torch.rand_like(layer.weight.data))
            layer.weight.data -= 0.5
            layer.weight.data *= 0.2
            if args.debug:
                print(layer.weight.data, file=sys.stderr)

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

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    trained_N = len(training_indices)
    reserved_N = len(reserved_indices)
    trained_error = []
    validated_error = []

    dWNorm = [ torch.empty(layer.weight.data.size()[0], 0) for layer in
               [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
    dTheta = [ torch.empty(layer.weight.data.size()[0], 0) for layer in
               [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]

    for epoch in range(args.epochs):
        if args.sgd:
            # Stochastic gradient descent: shuffle
            training_indices = torch.stack([training_indices[i] for i in torch.randperm(trained_N)])
        passed = 0
        failed = 0
        LW_i = [ layer.weight.data.clone().detach().requires_grad_(False) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        # Permute the order of the data for stochastic batch descent.
        reserved_X = X.index_select(0, reserved_indices)
        reserved_Y = Y.index_select(0, reserved_indices)

        batch = 0
        offset = 0
        trained_hits = 0
        while offset < trained_N:
            # Zero out gradients at the start of each batch.
            if args.autograd_backprop:
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    model.zero_grad()

            # Map training set input and target to the current batch based on pre-computed index order.
            batch_indices = training_indices[offset:offset+args.batch_size]
            batch_N = len(batch_indices)
            offset += args.batch_size
            batch_X = X.index_select(0, batch_indices)
            batch_Y = Y.index_select(0, batch_indices)

            # Train the model on the input X
            trained_Y = model(batch_X)
            # Make label predictions from argmax of the output.
            trained_labels = torch.stack([one_hots[y.argmax().item()] for y in trained_Y ])
            # Determine the count of hits and misses and update the training confusion matrix.
            trained_hits += batch_Y.eq(trained_labels).all(1).to(torch.int).sum()
            for i in range(len(batch_indices)):
                trained_confusion[batch_Y[i].argmax().item()][trained_Y[i].argmax().item()] +=1

            # Calculate losses for back-propagation.
            loss = loss_fn(trained_Y, batch_Y)
            loss.backward()

            # Backpropagation of loss gradients.
            if args.autograd_backprop:
                optimizer.step()
            else:
                # Turn off autograd so as to not pollute the gradients during back-prop.
                with torch.no_grad():
                    b_p = 0
                    for p in model.parameters():
                        p -= args.learning_rate * (1 - args.momentum) * p.grad + args.momentum * grad_bias[b_p]
                        grad_bias[b_p].copy_(p.grad)
                        b_p += 1

        # Test predictions on validation set.
        with torch.no_grad():
            validated_Y = model(reserved_X)
            validated_labels = torch.stack([one_hots[y.argmax().item()] for y in validated_Y])
            validated_hits = reserved_Y.eq(validated_labels).all(1).to(torch.int).sum()
            for i in range(len(reserved_indices)):
                validated_confusion[reserved_Y[i].argmax().item()][validated_Y[i].argmax().item()] +=1

        # Report epoch results.
        training_accuracy = float(trained_hits) / float(trained_N)
        trained_error.append(1.0 - training_accuracy)
        print("TRAIN[{}]: {}/{} ({})".format(epoch, trained_hits, trained_N, training_accuracy), file=sys.stderr)
        if args.debug:
            print_confusion_matrix(trained_confusion)

        validated_accuracy = float(validated_hits) / float(reserved_N)
        validated_error.append(1.0 - validated_accuracy)
        print("VALID[{}]: {}/{} ({})".format(epoch, validated_hits, reserved_N, validated_accuracy), file=sys.stderr)
        if args.debug:
            print_confusion_matrix(validated_confusion)

        # Calculate weight changes over the epoch.
        LW_j = [ layer.weight.data.clone().detach().requires_grad_(False) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        weight_change(LW_i, LW_j, dWNorm, dTheta)
        if (epoch % args.plot_every == 0):
            plot_training_validation_accuracy(model, epoch, trained_error, validated_error)
            plot_confusion_matrix(model, epoch, 'training', trained_confusion)
            plot_confusion_matrix(model, epoch, 'validation', validated_confusion)
            plot_weight_angle_changes(model, epoch, dTheta)
            plot_weight_magnitude_changes(model, epoch, dWNorm)
            plot_activation_heatmap(model, epoch, activations)
            if (args.interactive):
                show_combined_plots(epoch)

        # Reset confusion matrices for next epoch.
        trained_confusion.fill_(0)
        validated_confusion.fill_(0)

def show_combined_plots(epoch):
    return

def plot_training_validation_accuracy(model, epoch, trained_error, validated_error):
    mplp.plot(list(range(len(trained_error))), trained_error, 'r-')
    mplp.plot(list(range(len(validated_error))), validated_error, 'b-')
    mplp.legend(labels=('training', 'validation'), loc='upper right')
    mplp.xlabel('Training Epoch')
    mplp.ylabel('Classification Errors (%)')
    mplp.savefig("accuracy-{}.png".format(epoch))
    mplp.close()

def plot_weight_angle_changes(model, epoch, dTheta):
    layers = [n for n, c in model.named_children() if type(c) is torch.nn.Linear]
    i = 0
    for dtheta in dTheta:
        mplp.pcolormesh(dtheta.numpy(), cmap='hot')
        mplp.colorbar(label='Δθ')
        mplp.xlabel('Training Epoch')
        mplp.ylabel('{} Node'.format(layers[i]))
        mplp.savefig("weight-angle-changes-{}-{}.png".format(layers[i], epoch))
        mplp.close()
        i+=1

def plot_weight_magnitude_changes(model, epoch, dWNorm):
    layers = [n for n, c in model.named_children() if type(c) is torch.nn.Linear]
    i = 0
    for dwnorm in dWNorm:
        mplp.pcolormesh(dwnorm.numpy(), cmap='hot')
        mplp.colorbar(label='Δ‖w‖')
        mplp.xlabel('Training Epoch')
        mplp.ylabel('{} Node'.format(layers[i]))
        mplp.savefig("weight-magnitude-changes-{}-{}.png".format(layers[i], epoch))
        mplp.close()
        i += 1

def plot_confusion_matrix(model, epoch, which, confusion_matrix):
    mplp.imshow(confusion_matrix.numpy(), cmap='hot')
    mplp.colorbar(label='Predictions')
    mplp.xlabel('Predicted Class')
    mplp.ylabel('Target Class')
    mplp.savefig("confusion-matrix-{}-{}.png".format(which, epoch))
    mplp.close()

def print_confusion_matrix(M):
    s = ""
    for r in range(M.size()[0]):
        s += "|"
        for c in range(M.size()[1]):
            if M[r][c] == 0:
                mark = ' '
            else:
                mark = '+' if r == c else '-'
            s += "{}{}|".format(mark, M[r][c] if M[r][c] > 0 else ' ')
        s += "\n"
    print(s)

def plot_activation_heatmap(model, epoch, activations):
    for layer in activations:
        mplp.pcolormesh(activations[layer].numpy(), cmap='hot')
        mplp.colorbar(label='Activations')
        mplp.xlabel('Node Output')
        mplp.ylabel('{} Node'.format(layer))
        bin_ticks = [x for x in range(args.activation_bins + 1) if (x % (args.activation_bins/10)) == 0]
        mplp.xticks(bin_ticks, [ b / args.activation_bins for b in bin_ticks])
        mplp.savefig("activations-{}-{}.png".format(layer, epoch))
        mplp.close()

"""
Test NN.
"""
def test_nn(model, X, Y):
    classes = model[-1].out_features
    tested_confusion = torch.zeros(classes, classes)
    one_hots = torch.nn.functional.one_hot(torch.arange(0,classes)).type(torch.double)
    passed = 0
    failed = 0
    with torch.no_grad():
        predicted_Y = model(X)
        predicted_labels = torch.stack([one_hots[y.argmax().item()] for y in predicted_Y])
        predicted_hits = Y.eq(predicted_labels).all(1).to(torch.int).sum()
        for i in range(Y.size()[0]):
            tested_confusion[Y[i].argmax().item()][predicted_Y[i].argmax().item()] +=1

    print("TEST: {}/{}".format(predicted_hits, Y.size()[0]), file=sys.stderr)
    print_confusion_matrix(tested_confusion)

def capture_hidden_outputs_hook(module, features_in, features_out, **kwargs):
    for y in features_out:
        d = 0
        for f in y:
            f_bin = math.ceil(args.activation_bins * f) - 1
            activations[kwargs['name']][d][f_bin] += 1
            d += 1
    return None

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

# Set loss function to be mean squared error with summation over each training batch.
loss_fn = torch.nn.MSELoss(reduction='sum')

# Load input (in ARFF format).
X, Y, L = load_arff(args.arff)

# Build set of layers in order.
layers = OrderedDict()
activations = OrderedDict()
layers['P'] = torch.nn.LayerNorm(layer_D[0], elementwise_affine=False)
for i in range(len(layer_D)-1):
    linear_layer_name = 'L{}'.format(i)
    layers[linear_layer_name] = torch.nn.Linear(layer_D[i], layer_D[i+1]).to(torch.double)
    if i < len(layer_D) - 2:
        activation_layer_name = 'A{}'.format(i)
        layers[activation_layer_name] = activation_unit().to(torch.double)
        activations[activation_layer_name] = torch.zeros(layer_D[i+1], args.activation_bins).to(torch.double)
        layers[activation_layer_name].register_forward_hook(functools.partial(capture_hidden_outputs_hook, name=activation_layer_name))

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
