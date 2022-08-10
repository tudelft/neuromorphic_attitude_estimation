# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

###############################################################################
# Import packages
###############################################################################

from datetime import datetime
import yaml

from quaternions import inclination_loss, relative_inclination
import numpy as np
import torch 
import torch.nn as nn

from snn_model import SNN
from data import load_data_list, Dataset
from ea_training import CMAES

###############################################################################
# Define hyperparameters
###############################################################################

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running network on the following device: {device}')

if device == 'cuda':
    torch.cuda.empty_cache()

# data parameters
frequency = 100

# model parameters
snn_size = [60, 30, 30]
output_size = 4

# training parameters
train_seq_size = 5000
train_batch_size = 3
crit_type = 'SmoothL1Loss'
pop_size = 50
ngens = 3000

# validation parameters
val_batch_size = 10



###############################################################################
# Load data
###############################################################################

skip_initial = 6000 # skip the initial values to skip the liftoff in the dataset
normalization_type = 'minmax' # choose the normalization type (from minmax and standardize only currently)
gyro_max = 4.2 # absolute maximum value for normalization (these values are obtained from the used datasets)
acc_max = 16.8 # absolute maximum value for normalization (these values are obtained from the used datasets)

data_folder = f'/home/sstroobants/Opslag/owncloud/PhD/Code/IMU_NN/data/{frequency}hz' #this is not relative to cwd, depends on machine.
datasets = load_data_list(data_folder, skip_initial=skip_initial, normalization=normalization_type, gyro_max=gyro_max, acc_max=acc_max)
training_data = datasets[:-1]
val_data = [datasets[-1]]
test_data = datasets[-1]

train_dataset = Dataset(training_data, seq_length=train_seq_size, inter_seq_dist=100)
train_smplr = torch.utils.data.RandomSampler(np.arange(len(train_dataset)))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          shuffle=False,
                          batch_size=train_batch_size,
                          sampler=train_smplr,
                          drop_last=True)

val_dataset = Dataset(val_data, seq_length=train_seq_size, inter_seq_dist=100)
val_smplr = torch.utils.data.RandomSampler(np.arange(len(val_dataset)))
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                          shuffle=False,
                          batch_size=val_batch_size,
                          sampler=val_smplr,
                          drop_last=True)


###############################################################################
# Build model
###############################################################################

model = SNN(snn_size, output_size)
model = model.to(device)


###############################################################################
# Specify the loss function + optimizer
###############################################################################

# Loss and optimizer
if crit_type == 'SmoothL1Loss':
    criterion = nn.SmoothL1Loss()
elif crit_type == 'MSELoss':
    criterion = nn.MSELoss()
else:
    print("[ERROR] No valid criterion specified (choose from [SmoothL1Loss, MSELoss])")


def evaluate(output, target):
    q, q0 = inclination_loss(output, target)
    loss = criterion(q, q0)
    return loss.item()

###############################################################################
# Specify the Evolutionary Algorithm
###############################################################################

trainer = CMAES(model, evaluate, pop_size, device)

best_model, best_fitness, ngens, avg_time = trainer.fit(train_loader, val_loader, ngens)
print(f'Best obtained validation loss was : {best_fitness}')

###############################################################################
# Saving model to .pt file
###############################################################################

# filename for model storage
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")

model_save_file = f'models/snn{dt_string}.pt'

with open(model_save_file, 'wb') as f:
    torch.save(best_model, f)

hparams_save_file = f'runs/snn{dt_string}.txt'

hparams = {
            "best_loss": float(best_fitness),
            "avg_gen_calc_time": avg_time,
            "pop_size": pop_size,
            "number_generations": ngens, 
            "snn_size": snn_size, 
            "output_size": output_size, 
            "train_seq_size": train_seq_size, 
            "train_batch_size": train_batch_size,
            "crit_type": crit_type,
            "frequency": frequency,
            "normalization_type": normalization_type,
            "gyro_max": gyro_max,
            "acc_max": acc_max,
            "skip_initial": skip_initial      
}

with open(hparams_save_file, 'w') as f:
    yaml.dump(hparams, f, sort_keys=False)