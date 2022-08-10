from os import stat
import random
from time import time
import yaml
from argparse import ArgumentParser
from itertools import repeat

import ahrs
import matplotlib.pyplot as plt
import numpy as np
import torch

from multiprocessing import Pool

from datamodule.data_loader import load_datasets

from utils.quaternions import (from_euler_angles_numpy, inclination_loss, relative_angle,
                         relative_inclination, to_euler_angles,
                         to_euler_angles_numpy)
from utils.network_utils import state_dict_to_weights_array

class PSO:
    def __init__(self, n_variables, freq, min_func, n_particles=70):
        self.n_particles = n_particles
        self.X = np.random.rand(self.n_particles, n_variables) @ np.array([[0.8, 0], [0, 0.2]]) + np.array([0.2, 0])
        # self.X = np.random.rand(self.n_particles, n_variables) * 0.5 + 0.005
        self.V = np.random.randn(self.n_particles, n_variables) * 0.05
        self.freq = freq
        self.min_func = min_func

        self.c1 = 0.15
        self.c2 = 0.05
        self.w = 0.8

    def calc_initial_angle(self, eulers):
        # Calculate initial angle for filters
        eulers_t0 = np.array([[eulers[0][0], eulers[0][1], 0]])
        q0 = from_euler_angles_numpy(eulers_t0)[0]
        q0 = [0, 0, 0, 1]
        return q0
    
    def calc_output(self, gain, inputs, eulers):
        err = 0
        for i in range(inputs.size()[0]):
            # print(i)
            err += self.obj(gain, inputs[i, :, :], eulers[:, i, :])
        return err

    def initial_call(self, inputs, eulers):
        self.pbest = self.X
        self.pbest_obj = []

        with Pool() as pool:
            self.pbest_obj = pool.starmap(self.calc_output, zip(self.pbest, repeat(inputs), repeat(eulers)))
        
        self.gbest = self.pbest[np.argmin(self.pbest_obj), :]
        self.gbest_obj = np.min(self.pbest_obj)


    def obj(self, gain, inputs, eulers):
        if (np.all(gain) > 0.0001) and (np.all(gain < 0.9999)):
            gyr_np = inputs.numpy()[:, :3]
            acc_np = inputs.numpy()[:, 3:6]
            q0 = self.calc_initial_angle(eulers)
            # output = self.min_func(gyr=gyr_np, acc=acc_np, frequency=self.freq, q0=q0, gain=gain)
            output = self.min_func(gyr=gyr_np, acc=acc_np, frequency=self.freq, q0=q0, k_P=gain[0], k_I=gain[1])
            output_eulers = to_euler_angles_numpy(output.Q)
            err = np.sqrt((eulers - output_eulers[:, :2])**2).mean()
        else:
            err = 2
        return err

    def update(self, inputs, eulers):
        r1, r2 = np.random.rand(2)
        self.V = self.w * self.V + self.c1 * r1 * (self.pbest - self.X) + self.c2 * r2 * (self.gbest - self.X)
        self.X = self.X + self.V
        self.X[self.X > 1] = 1
        self.X[self.X < 0] = 0
        # obj = []
        # for gain in self.X:
        #     err = 0
        #     for i in range(inputs.size()[0]):
        #         # print(inputs[i, :, :])
        #         # print(i)
        #         err += self.obj(gain, inputs[i, :, :], eulers[:, i, :])
        #     obj.append(err)
        # obj = np.array(obj)
        with Pool() as pool:
            obj = pool.starmap(self.calc_output, zip(self.X, repeat(inputs), repeat(eulers)))

        self.pbest[(self.pbest_obj >= obj), :] = self.X[(self.pbest_obj >= obj), :]
        self.pbest_obj = np.array([self.pbest_obj, obj]).min(axis=0)
        self.gbest = self.pbest[np.argmin(self.pbest_obj), :]
        self.gbest_obj = np.min(self.pbest_obj)
        # print(self.X)
        print(self.gbest, self.gbest_obj)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="runs/snn_backprop28022022_232433")
    parser.add_argument("--plot_eul", type=bool, default=False)
    args = parser.parse_args()

    # Config from yaml file
    with open(f'{args.output_folder}/config.txt', "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    g_madgwick = torch.Generator()
    # g_madgwick.seed()
    g_madgwick.manual_seed(529812334)

    # Set batch to 1 for evaluation
    config["data"]["train_batch_size"] = 2
    config["data"]["val_batch_size"] = 1
    config["data"]["seq_length"] = 5000
    
    config["data"]["encoding"] = None
    config["data"]["normalization"] = None
    train_loader, val_loader = load_datasets(config, generator=g_madgwick)

    # Load model from file
    device = torch.device('cpu')

    
    optim = PSO(2, config["data"]["frequency"], ahrs.filters.Mahony)
    input, target = next(iter(train_loader))
    # print(input)
    target = target.permute(1, 0, 2)
    eulers = target.squeeze(1).numpy()
    optim.initial_call(input, eulers)
    N_GENS = 50

    for i in range(N_GENS):
        g_madgwick.manual_seed(529812334)
        input, target = next(iter(train_loader))
        # print(input)
        target = target.permute(1, 0, 2)
        eulers = target.squeeze(1).numpy()
        optim.update(input, eulers)