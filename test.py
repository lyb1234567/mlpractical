from mlp.layers import Layer
import numpy as np

class MaxPoolingLayer(Layer):

    def __init__(self, pool_size=2):
        """Construct a new max-pooling layer.

        Args:
            pool_size: Positive integer specifying size of pools over
               which to take maximum value. The outputs of the layer
               feeding in to this layer must have a dimension which
               is a multiple of this pool size such that the outputs
               can be split in to pools with no dimensions left over.
        """
        assert pool_size > 0
        self.pool_size = pool_size

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        This corresponds to taking the maximum over non-overlapping pools of
        inputs of a fixed size `pool_size`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        assert inputs.shape[-1] % self.pool_size == 0, (
            'Last dimension of inputs must be multiple of pool size')
        outputs = []
        self.mask = []
        for i in range(inputs.shape[0]):
            count = inputs.shape[-1] // self.pool_size
            temp = inputs[i]
            initial = 0
            sub = []
            sub_mask = []
            for j in range(count):
                sub.append(np.max(temp[initial:initial + self.pool_size]))
                sub_mask.append(np.where(temp == np.max(temp[initial:initial + self.pool_size]))[0][0])
                initial = initial + self.pool_size
            self.mask.append(sub_mask)
            outputs.append(sub)
        outputs = np.array(outputs)
        self.mask = np.array(self.mask)
        final_mask = np.zeros(inputs.shape)
        for i in range(self.mask.shape[0]):
            temp = self.mask[i]
            for pos in temp:
                final_mask[i][pos] = 1
        self.mask = final_mask
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        brop_outputs = self.mask
        for k in range(inputs.shape[0]):
            sub_mask = self.mask[k]
            count = 0
            for j in range(inputs.shape[-1]):
                if sub_mask[j] == 1:
                    brop_outputs[k][j] = grads_wrt_outputs[k][count]
                    count = count + 1
        return brop_outputs

    def __repr__(self):
        return 'MaxPoolingLayer(pool_size={0})'.format(self.pool_size)
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
from mlp.models import MultipleLayerModel
from mlp.layers import AffineLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import MomentumLearningRule
from mlp.optimisers import Optimiser
import matplotlib.pyplot as plt

# Seed a random number generator
seed = 31102016
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]
#
# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

#
# Size of pools to take maximum over
pool_size = 2

input_dim, output_dim, hidden_dim = 784, 10, 100

# Use Glorot initialisation scheme for weights and zero biases
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

# Create three affine layer model interleaved with max-pooling layers
model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim * pool_size, weights_init, biases_init),
    MaxPoolingLayer(pool_size),
    AffineLayer(hidden_dim, hidden_dim * pool_size, weights_init, biases_init),
    MaxPoolingLayer(pool_size),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

# Multiclass classification therefore use cross-entropy + softmax error
error = CrossEntropySoftmaxError()

# Use a momentum learning rule - you could use an adaptive learning rule
# implemented for the coursework here instead
learning_rule = MomentumLearningRule(0.02, 0.9)

# Monitor classification accuracy during training
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(
    model, error, learning_rule, train_data, valid_data, data_monitors)

num_epochs = 2
stats_interval = 5

stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

# Plot the change in the validation and training set error over training.
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
              stats[1:, keys[k]], label=k)
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')

# Plot the change in the validation and training set accuracy over training.
fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
              stats[1:, keys[k]], label=k)
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')