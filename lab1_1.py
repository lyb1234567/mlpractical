
import numpy as np
import sys
sys.path.append('/path/to/mlpractical')
import os
import matplotlib.pyplot as plt
import mlp.data_providers as data_providers
mnist_dp = data_providers.MNISTDataProvider(
    which_set='valid', batch_size=500, max_num_batches=1, shuffle_order=True)
inputs=[]
targets=[]
for i,j in mnist_dp:
    inputs=i
    targets=j
def lab1_1(inputs,target):
    batch=inputs.shape[0]//100
    image_batch=[]
    for i in range(batch):
        batch_slice=slice(i,i+100)
        sub=inputs[batch_slice]
        image_batch.append(sub)
        i=i+100
    return image_batch
def show_batch_of_images(img_batch):
    for batch in img_batch:

        f, axarr = plt.subplots(10, 10)
        count=0
        for i in range(10):
             for j in range(10):
                sub_slice=slice(count,count+1)
                sub=batch[sub_slice]
                sub=sub.reshape(28,28)
                axarr[i,j].imshow(sub)
                count=count+1
        plt.figure()

    plt.show()
img_batch=lab1_1(inputs,targets)
show_batch_of_images(img_batch)

