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
import os.path

"""
Command line arguments
"""
parser = argparse.ArgumentParser(description='Parameterized Interactive Neural Network Iteratively Plotting Experimental Decisions')
parser.add_argument('--train-arff', type=str)
parser.add_argument('--test-arff', type=str)

parser.add_argument('--model-file', type=str, default='model.nn')
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
parser.add_argument('--workspace-dir', type=str, default='.')

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
def train_nn(model, X, Y, test_X, test_Y):
    classes = model[-1].out_features
    sample_count = X.size()[0]
    trained_confusion = torch.zeros(classes, classes).type(torch.int)
    validated_confusion = torch.zeros(classes, classes).type(torch.int)
    tested_confusion = torch.zeros(classes, classes).type(torch.int)
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
    tested_N = len(test_Y)
    trained_error = []
    validated_error = []
    tested_error = []
    indices = {}

    hits = {}
    accuracy = {}

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
        training_X = X.index_select(0, training_indices)
        training_Y = Y.index_select(0, training_indices)

        while offset < trained_N:
            # Zero out gradients at the start of each batch.
            if args.autograd_backprop:
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    model.zero_grad()

            # Map training set input and target to the current batch based on pre-permuted index order.
            batch_indices = training_indices[offset:offset+args.batch_size]
            batch_X = X.index_select(0, batch_indices)
            batch_Y = Y.index_select(0, batch_indices)

            # Train the model on the input X
            trained_Y = model(batch_X)

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
                        grad_bias[b_p].copy_(args.learning_rate * (1 - args.momentum) * p.grad + args.momentum * grad_bias[b_p])
                        p -= grad_bias[b_p]
                        b_p += 1
            offset += args.batch_size

        # Test predictions on training set at the end of the epoch. Gains or losses between batches within an epoch are not captured.
        trained_hits, trained_accuracy = test_batch(model, training_X, training_Y, trained_confusion)
        # Test predictions on validation set.
        validated_hits, validated_accuracy = test_batch(model, reserved_X, reserved_Y, validated_confusion)
        # Test predictions on test set if one has been supplied.
        if test_X is not None and test_Y is not None:
            tested_hits, tested_accuracy = test_batch(model, test_X, test_Y, tested_confusion)

        # Report epoch results.
        trained_error.append(1.0 - trained_accuracy)
        print("TRAIN[{}]: {}/{} ({})".format(epoch, trained_hits, trained_N, trained_accuracy), file=sys.stderr)
        if args.debug:
            print_confusion_matrix(trained_confusion)

        validated_error.append(1.0 - validated_accuracy)
        print("VALID[{}]: {}/{} ({})".format(epoch, validated_hits, reserved_N, validated_accuracy), file=sys.stderr)
        if args.debug:
            print_confusion_matrix(validated_confusion)

        if test_X is not None and test_Y is not None:
            tested_accuracy = float(tested_hits) / float(tested_N)
            tested_error.append(1.0 - tested_accuracy)
            print("TEST[{}]: {}/{} ({})".format(epoch, tested_hits, tested_N, tested_accuracy), file=sys.stderr)
            if args.debug:
                print_confusion_matrix(validated_confusion)


        # Calculate weight changes over the epoch.
        LW_j = [ layer.weight.data.clone().detach().requires_grad_(False) for layer in
                 [ layer for layer in model.children() if type(layer) is torch.nn.Linear ] ]
        weight_change(LW_i, LW_j, dWNorm, dTheta)
        if ((epoch + 1) % args.plot_every == 0):
            plot_training_validation_accuracy(model, epoch, trained_error, validated_error, tested_error)
            plot_confusion_matrix(model, epoch, 'training', trained_confusion)
            plot_confusion_matrix(model, epoch, 'validation', validated_confusion)
            if args.test_arff is not None:
                plot_confusion_matrix(model, epoch, 'test', tested_confusion)
            plot_weight_angle_changes(model, epoch, dTheta)
            plot_weight_magnitude_changes(model, epoch, dWNorm)
            plot_activation_heatmap(model, epoch, activations)
            if (args.interactive):
                show_combined_plots(epoch)

        # Reset confusion matrices for next epoch.
        trained_confusion.fill_(0)
        validated_confusion.fill_(0)
        tested_confusion.fill_(0)

def show_combined_plots(epoch):
    return

def plot_training_validation_accuracy(model, epoch, trained_error, validated_error, tested_error):
    mplp.plot(list(range(len(trained_error))), trained_error, 'r-')
    mplp.plot(list(range(len(validated_error))), validated_error, 'b-')
    legend_labels=('training', 'validation')
    if len(tested_error) != 0:
        legend_labels += ('test',)
        mplp.plot(list(range(len(tested_error))), tested_error, 'g-')
    mplp.legend(labels=legend_labels, loc='upper right')
    mplp.xlabel('Training Epoch')
    mplp.ylabel('Classification Errors (%)')
    mplp.title('Model Accuracy over Time\n{}'.format(model_params_shorthand))
    mplp.savefig(os.path.join(args.workspace_dir, 'accuracy-{}.png'.format(epoch)))
    mplp.close()

def plot_weight_angle_changes(model, epoch, dTheta):
    layers = [n for n, c in model.named_children() if type(c) is torch.nn.Linear]
    i = 0
    for dtheta in dTheta:
        mplp.pcolormesh(dtheta.numpy(), cmap='hot')
        mplp.colorbar(label='Δθ')
        mplp.xlabel('Training Epoch')
        mplp.ylabel('{} Node'.format(layers[i]))
        mplp.title('Weight Vector Angle Changes Per {} Node Over Time\n{}'.format(layers[i], model_params_shorthand))
        mplp.savefig(os.path.join(args.workspace_dir, 'weight-angle-changes-{}-{}.png'.format(layers[i], epoch)))
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
        mplp.title('Weight Vector Norm Changes Per {} Node Over Time\n{}'.format(layers[i], model_params_shorthand))
        mplp.savefig(os.path.join(args.workspace_dir, 'weight-magnitude-changes-{}-{}.png'.format(layers[i], epoch)))
        mplp.close()
        i += 1

def plot_confusion_matrix(model, epoch, which, confusion_matrix):
    mplp.imshow(confusion_matrix.numpy(), cmap='hot')
    mplp.colorbar(label='Predictions')
    mplp.xlabel('Predicted Class')
    mplp.ylabel('Target Class')
    mplp.title('{} Set Confusion Matrix @ t = {}\n{}'.format(which.capitalize(), epoch, model_params_shorthand))
    mplp.savefig(os.path.join(args.workspace_dir, 'confusion-matrix-{}-{}.png'.format(which, epoch)))
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
        mplp.title('Non-linear Activations Per {} Node\n{}'.format(layer, model_params_shorthand))
        mplp.savefig(os.path.join(args.workspace_dir, 'activations-{}-{}.png'.format(layer, epoch)))
        mplp.close()

def test_batch(model, X_in, Y_tru, M_confusion):
    with torch.no_grad():
        Y_out = model(X_in)
        Y_lbl = torch.stack([one_hots[y.argmax().item()] for y in Y_out])
        hits = Y_tru.eq(Y_lbl).all(1).to(torch.int).sum()
        for i in range(Y_tru.size()[0]):
            M_confusion[Y_tru[i].argmax().item()][Y_out[i].argmax().item()] += 1
    return (hits, float(hits) / float(Y_tru.size()[0]))

"""
Test NN.
"""
def test_nn(model, X, Y):
    M_confusion = torch.zeros(model[-1].out_features, model[-1].out_features)
    hits, accuracy = test_batch(model, X, Y, M_confusion)
    print("TEST: {}/{} ({})".format(hits, Y.size()[0], accuracy ), file=sys.stderr)
    if args.debug:
        print_confusion_matrix(M_confusion)

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

os.makedirs(args.workspace_dir, mode=0o770, exist_ok=True)

# Get layer dims.
layer_D = [ int(D) for D in args.layer_dims.split(",") ]

if len(layer_D) < 2:
    raise argparse.ArgumentError("Must define at least two layer dimensions (input + output)")

# Determine activation unit.
activation_unit = activation_units[args.activation_unit]

# Create shorthand string to describe model params (useful in plots).
model_params_shorthand = '(i = {}, h = {}, o = {}, a={}, lr={}, bs={}, α={})'.format(layer_D[0], layer_D[1:-1], layer_D[-1], args.activation_unit, args.learning_rate, args.batch_size, args.momentum)

# Set loss function to be mean squared error with summation over each training batch.
loss_fn = torch.nn.MSELoss(reduction='sum')

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
        if args.train_arff is not None:
            activations[activation_layer_name] = torch.zeros(layer_D[i+1], args.activation_bins).to(torch.double)
            layers[activation_layer_name].register_forward_hook(functools.partial(capture_hidden_outputs_hook, name=activation_layer_name))

model = torch.nn.Sequential(layers)
one_hots = torch.nn.functional.one_hot(torch.arange(0,model[-1].out_features)).type(torch.double)

# Load training input (in ARFF format).
if args.train_arff is not None:
    X_train, Y_train, L_train = load_arff(args.train_arff)

X_test = None
Y_test = None
L_test = None

# Load testing input (in ARFF format).
if args.test_arff is not None:
    X_test, Y_test, L_test = load_arff(args.test_arff)

# Create NN model.
print(model, file=sys.stderr)

# Train or test, depending on user spec.
# @TODO: Save trained model for future loading. Otherwise, have to train before testing.
if args.train_arff is not None:
    train_nn(model, X_train, Y_train, X_test, Y_test)
    with open(os.path.join(args.workspace_dir, args.model_file), 'wb') as fout:
        pickle.dump(model.state_dict(), fout)
elif args.test_arff is not None:
    with open(os.path.join(args.workspace_dir, args.model_file), 'rb') as fin:
        model.load_state_dict(pickle.load(fin))
    test_nn(model, X_test, Y_test)
