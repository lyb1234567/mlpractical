import mlp.data_providers as data_providers
import matplotlib.pyplot as plt
import numpy as np
batch_size = 3
for window_size in [2, 5, 10]:
    met_dp = data_providers.MetOfficeDataProvider(
        window_size=window_size, batch_size=batch_size,
        max_num_batches=1, shuffle_order=False)
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.set_title('Window size {0}'.format(window_size))
    ax.set_xlabel('Day in window')
    ax.set_ylabel('Normalised reading')
    # iterate over data provider batches checking size and plotting
    for inputs, targets in met_dp:
        print(inputs)
        assert inputs.shape == (batch_size, window_size - 1)
        assert targets.shape == (batch_size, )
        ax.plot(np.c_[inputs, targets].T, '.-')
        ax.plot([window_size - 1] * batch_size, targets, 'ko')
    print("Passed !")
    plt.show()