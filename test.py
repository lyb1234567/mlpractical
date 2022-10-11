import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/admin/Desktop/mlpractical')
from mpl_toolkits.mplot3d import Axes3D
from mlp.data_providers import CCPPDataProvider
data_provider = CCPPDataProvider(
    which_set='train',
    input_dims=[0, 1],
    batch_size=5000,
    max_num_batches=1,
    shuffle_order=False
)
inputs, targets = data_provider.next()
from mlp.layers import AffineLayer
from mlp.errors import SumOfSquaredDiffsError
from mlp.models import SingleLayerModel
from mlp.initialisers import UniformInit, ConstantInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser
import logging

# Seed a random number generator
seed = 27092016
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the CCPP training set
train_data = CCPPDataProvider('train', [0, 1], batch_size=100, rng=rng)
input_dim, output_dim = 2, 1

# Create a parameter initialiser which will sample random uniform values
# from [-0.1, 0.1]
param_init = UniformInit(-0.1, 0.1, rng=rng)
# Create our single layer model
layer = AffineLayer(input_dim, output_dim, param_init, param_init)
model = SingleLayerModel(layer)
# Initialise the error object
error = SumOfSquaredDiffsError()

# Use a basic gradient descent learning rule with a small learning rate
learning_rule = GradientDescentLearningRule(learning_rate=1e-2)

# Use the created objects to initialise a new Optimiser instance.
optimiser = Optimiser(model, error, learning_rule, train_data)

# Run the optimiser for 5 epochs (full passes through the training set)
# printing statistics every epoch.
stats, keys = optimiser.train(num_epochs=100, stats_interval=1)

data_provider = CCPPDataProvider(
    which_set='train',
    input_dims=[0, 1],
    batch_size=5000,
    max_num_batches=1,
    shuffle_order=False
)

inputs, targets = data_provider.next()

# Calculate predicted model outputs
outputs = model.fprop(inputs)[-1]
# Plot target and predicted outputs against inputs on same axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(inputs[:, 0], inputs[:, 1], targets[:, 0], 'r.', ms=2)
ax.plot(inputs[:, 0], inputs[:, 1], outputs[:, 0], 'b.', ms=2)
ax.set_xlabel('Input dim 1')
ax.set_ylabel('Input dim 2')
ax.set_zlabel('Output')
ax.legend(['Targets', 'Predictions'], frameon=False)
fig.tight_layout()