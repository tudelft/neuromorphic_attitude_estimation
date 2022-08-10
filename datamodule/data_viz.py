# This module should contain the following functionality:

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_batch(imu, target):

    inter_neuron_dist = 1.5
    samples = imu.size()[0]
    nbins = int(imu.size()[1] / 6)
    time = np.arange(samples)
    ntargets = int(target.size()[1])

    plt.figure()
    for i in range(6):
        plt.subplot(6, 1, i+1)
        for j in range(nbins):
            imu[..., i*nbins + j][imu[..., i*nbins + j] == 0] = np.nan
            plt.plot(time, imu[..., i*nbins + j] + j*inter_neuron_dist, 'b.')
            plt.plot(time, np.zeros(samples) + j*inter_neuron_dist, 'k')

    plt.figure()
    for i in range(ntargets):
        plt.subplot(ntargets, 1, i+1)
        plt.plot(time, target[:, i])
    plt.show()