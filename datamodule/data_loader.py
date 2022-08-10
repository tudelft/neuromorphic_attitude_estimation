import os

import torch
import pandas as pd
import numpy as np

from utils.quaternions import to_euler_angles, multiply_quat, conj_quat

# This module should contain the following functionality:

# A custom torch dataset that can be used to sample a random subsample from a list of datasets. 
# This should have the possibility to rotate the dataset around the yaw

class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_folder, options):
        self.options = options

        # Read all datasets in the given training folder
        self.dataset_list = []
        for (_, _, filenames) in os.walk(training_folder, topdown=True):
            for filename in filenames:
                self.dataset_list.append(pd.read_csv(os.path.join(training_folder, filename)))
        self.seq_length = options['seq_length']
        inter_seq_dist = options['inter_seq_dist']

        # Create a list of custom (tuple) indices
        self.indices = []
        for i_dataset, dataset in enumerate(self.dataset_list):
            for idx in range(0, len(dataset) - self.seq_length, inter_seq_dist):
                self.indices.append((i_dataset, idx))

        # Define the input columns for the DataFrame
        if options['encoding'] != None:
            self.input_columns = [f'n{i}' for i in range(options['nbins'])]
        elif options['normalization'] != None:
            self.input_columns = ['gyro_x_norm', 'gyro_y_norm', 'gyro_z_norm',
                                  'acc_x_norm', 'acc_y_norm', 'acc_z_norm']
        else:
            self.input_columns = ['gyro_x', 'gyro_y', 'gyro_z',
                                  'acc_x', 'acc_y', 'acc_z']

        # Define the output columns for the DataFrame
        if options['output'] == 'eulers':
            self.output_columns = ['roll', 'pitch']
        elif options['output'] == 'eulers+vel':
            self.output_columns = ['roll', 'pitch', 'roll_vel', 'pitch_vel']
        elif options['output'] == 'quaternions':
            self.output_columns = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
        elif options['output'] == 'control':
            self.output_columns = ['torque_pitch', 'torque_roll']
        # self.input_columns = self.output_columns
# 
        print(f'loaded {len(self.dataset_list)} datasets with a total of {len(self.indices)} sequences')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)

    def __getitem__(self, idx):
        '''Generates a sequence of data with length seq_length'''
        eulers = None

        i_dataset, i_start = self.indices[idx]
        imu = torch.from_numpy(self.dataset_list[i_dataset].loc[i_start:i_start+self.seq_length - 1, self.input_columns].to_numpy())
        imu = imu.to(dtype=torch.float)
        target = torch.from_numpy(self.dataset_list[i_dataset].loc[i_start:i_start+self.seq_length - 1, self.output_columns].to_numpy())
        target = target.to(dtype=torch.float)

        # Rotate yaw if necessary
        if self.options['rotate_yaw']:
            if self.options['output'] == 'quaternion':
                eulers_t0 = to_euler_angles(target[0].unsqueeze(-2).unsqueeze(-2))
                yaw_t0 = eulers_t0[..., 2].item()
                rotation_tensor = torch.tensor([[[torch.cos(yaw_t0/2), 0., 0., torch.sin(yaw_t0/2) ]]]).to(target.device)
                target = multiply_quat(conj_quat(rotation_tensor), target)
            if self.options['output'] == 'eulers':
                pass #currently yaw is not considered, but otherwise this should be added
        else:
            return imu, target.squeeze(0)

def load_datasets(config, generator=None):
    
    train_folder = os.path.join(config["data"]["dir"], "Train")
    validation_folder = os.path.join(config["data"]["dir"], "Validation")

    train_dataset = Dataset(train_folder, config["data"])
    train_smplr = torch.utils.data.RandomSampler(np.arange(len(train_dataset)), generator=generator)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                            shuffle=False,
                            batch_size=config["data"]["train_batch_size"],
                            sampler=train_smplr,
                            drop_last=True)

    val_dataset = Dataset(validation_folder, config["data"])
    val_smplr = torch.utils.data.RandomSampler(np.arange(len(val_dataset)), generator=generator)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=config["data"]["val_batch_size"],
                            sampler=val_smplr,
                            drop_last=True)
    
    return train_loader, val_loader