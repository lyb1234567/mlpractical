# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
#
# def train_model_and_plot_stats(
#         model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
#     # As well as monitoring the error over training also monitor classification
#     # accuracy i.e. proportion of most-probable predicted classes being equal to targets
#     data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
#
#     # Use the created objects to initialise a new Optimiser instance.
#     optimiser = Optimiser(
#         model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)
#
#     # Run the optimiser for num_epochs epochs (full passes through the training set)
#     # printing statistics every epoch.
#     stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
#
#     # Plot the change in the validation and training set error over training.
#     fig_1 = plt.figure(figsize=(8, 4))
#     ax_1 = fig_1.add_subplot(111)
#     for k in ['error(train)', 'error(valid)']:
#         ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval,
#                   stats[1:, keys[k]], label=k)
#     ax_1.legend(loc=0)
#     ax_1.set_xlabel('Epoch number')
#     ax_1.set_ylabel('Error')
#
#     # Plot the change in the validation and training set accuracy over training.
#     fig_2 = plt.figure(figsize=(8, 4))
#     ax_2 = fig_2.add_subplot(111)
#     for k in ['acc(train)', 'acc(valid)']:
#         ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval,
#                   stats[1:, keys[k]], label=k)
#     ax_2.legend(loc=0)
#     ax_2.set_xlabel('Epoch number')
#     ax_2.set_xlabel('Accuracy')
#
#     return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2
#
# # The below code will set up the data providers, random number
# # generator and logger objects needed for training runs. As
# # loading the data from file take a little while you generally
# # will probably not want to reload the data providers on
# # every training run. If you wish to reset their state you
# # should instead use the .reset() method of the data providers.
# import sys
# sys.path.append('C:/Users/admin/Desktop/mlpractical')
# import numpy as np
# import logging
# from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
#
# # Seed a random number generator
# seed = 11102019
# rng = np.random.RandomState(seed)
# batch_size = 100
# # Set up a logger object to print info about the training run to stdout
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.handlers = [logging.StreamHandler()]
#
# # Create data provider objects for the MNIST data set
# train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
# valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
#
# from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
# from mlp.errors import CrossEntropySoftmaxError
# from mlp.models import MultipleLayerModel
# from mlp.initialisers import ConstantInit, GlorotUniformInit
# from mlp.learning_rules import AdamLearningRule
# from mlp.optimisers import Optimiser
#
# # Setup hyperparameters
# learning_rate = 0.001
# num_epochs = 100
# stats_interval = 1
# input_dim, output_dim, hidden_dim = 784, 47, 128
#
# weights_init = GlorotUniformInit(rng=rng)
# biases_init = ConstantInit(0.)
#
# # Create model with ONE hidden layer
# model = MultipleLayerModel([
#     AffineLayer(input_dim, hidden_dim, weights_init, biases_init), # hidden layer
#     ReluLayer(),
#     AffineLayer(hidden_dim, output_dim, weights_init, biases_init) # output layer
# ])
#
# error = CrossEntropySoftmaxError()
# # Use a Adam learning rule
# learning_rule = AdamLearningRule(learning_rate=learning_rate)
#
# # Remember to use notebook=False when you write a script to be run in a terminal
# _ = train_model_and_plot_stats(
#     model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)
import numpy as np
seed = 22102017
rng = np.random.RandomState(seed)
print(rng.uniform(size=(1,)+(3,4,5)))