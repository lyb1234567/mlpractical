import numpy as np
import matplotlib.pyplot as plt
import mlp.data_providers as data_providers
mnist_dp = data_providers.MNISTDataProvider(
    which_set='valid', batch_size=6, max_num_batches=10, shuffle_order=False)

for inputs, targets in mnist_dp:
    # Check that values are either 0 or 1
    assert np.all(np.logical_or(targets == 0., targets == 1.))
    # Check that there is exactly a single 1
    assert np.all(targets.sum(-1) == 1.)
    print(targets.shape)