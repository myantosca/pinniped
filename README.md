# PINNIPED (Parameterized Interactive Neural Network Iteratively Plotting Experimental Decisions)

# Usage

```
pinniped.py [-h] [--train-arff TRAIN_ARFF] [--test-arff TEST_ARFF]
                 [--model-file MODEL_FILE] [--epochs EPOCHS]
                 [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                 [--momentum MOMENTUM]
                 [--activation-unit {sigmoid,tanh,relu}] --layer-dims
                 LAYER_DIMS
                 [--reserve-frac RESERVE_FRAC | --reserve-every RESERVE_EVERY]
                 [--autograd-backprop] [--sgd] [--interactive] [--debug]
                 [--activation-bins ACTIVATION_BINS]
                 [--plot-every PLOT_EVERY]
                 [--plot-confusion-every PLOT_CONFUSION_EVERY]
                 [--workspace-dir WORKSPACE_DIR] [--save-every SAVE_EVERY]
                 [--profile]
```

`--train-arff TRAIN_ARFF`

	The filename of the ARFF holding the training data

`--test-arff TEST_ARFF`

	The filename of the ARFF holding the test data.
	If specified together with a training ARFF, test error also will be plotted.

`--model-file MODEL_FILE`

	The filename for saving the model during training or loading it for testing.

`--epochs EPOCHS`

	The number of iterations over the entire training set in which to train.

`--batch-size BATCH_SIZE`

	The number of samples ingested for training before learning occurs.

`--learning-rate LEARNING_RATE`

	The rate at which adjustments are made to the layer weight vectors.

`--momentum MOMENTUM`

    A factor for dampening the effect of learning.

`--activation-unit {sigmoid,tanh,relu}`

	The non-linear activation step function for hidden layer outputs.

`--layer-dims LAYER_DIMS`

	The dimensions of each layer, including input and output.
	The list of dimensions is specified as a comma-separated list of integers.

`--reserve-frac RESERVE_FRAC`

	The fraction of the training set to reserve for validation.
	This does a simple partition of the last samples by the given fraction.

`--reserve-every RESERVE_EVERY`

	Another way to specify samples to reserve for validation.
	This holds out every RESERVE_EVERY sample. Use this if the class labels
	are presented in order during training to ensure a similar distribution
	in the validation set.

`--autograd-backprop`

	Flag to use PyTorch optimizers for backpropagation.

`--sgd`

	Flag to enable stochastic gradient descent (SGD).
	This permutes the order of training samples prior to partitioning
	for validation, but the boundary between validation	and training is
	preserved after the first cut to avoid set bleed.
	Training order is permuted after the first cut, but validation
	order is not since it does not contribute to the learning.

`--interactive`

	TBD. This initially was meant to allow the user to pause between
	training epochs and view graphs at a glance but was backlogged
	to permit development of other features.

`--debug`

	Flag to provide some additional debug output to stderr.

`--activation-bins ACTIVATION_BINS`

	The activation histogram resolution. By default, 20.

`--plot-every PLOT_EVERY`

	Determines the plot cadence. For long-term training, plotting every
	epoch is time-consuming and often redundant. This allows the user
	to control the rate at which intermediate graphs are plotted.
	Default is 1.

`--plot-confusion-every PLOT_CONFUSION_EVERY`

	For long runs, many of the graphs do not benefit from intermediate
	plots since the final plot contains all the information preceding it.
	Confusion matrices are an exception since they do not exhibit history.
	This flag permits more frequent plotting of confusion matrices,
	even when the general plots are infrequent. Default is 1.

`--workspace-dir WORKSPACE_DIR`

	The directory prefix for model and plot output.
	Let's stay organized!

`--save-every SAVE_EVERY`

	Determines the cadence at which intermediate models are saved.
	Because sometimes you don't care about the 49th iteration in
	a 256 epoch run. Default is 1, i.e., every intermediate model.

`--profile`

	Flag to do basic execution profiling and dump a profile report
	at the end of a training or test run.

## Comments

PINNIPED is based on the PyTorch framework. It provides a simple means for
training a somewhat general feedforward neural network with backpropagation
and examining the intermediate stages of model evolution.

The program is designed with a mind toward ingesting audio features as
1D input samples and performs some basic preprocessing normalization prior
to admitting the input to the neural network proper.

## Known Issues

- PINNIPED is not truly interactive (yet). Time constraints prevented developing
a nice interface, but users can generate intermediate plots and model states
through the program's extensive set of command line arguments.
- Development uncovered a fair amount of precision issues and FP overflow.
Mitigations have been put in place for many of these, but some still exist.
Issues like catastrophic cancellation are bound to crop up when considering
input dimensions like 1,024 if one is not supremely careful.
- PINNIPED is _slow_. Performance optimizations were not as much of a focus
as parameterization in the initial assay.

## Contact

Please contact Michael Yantosca via e-mail at mike@archivarius.net for
all comments, questions, and bug reports.
