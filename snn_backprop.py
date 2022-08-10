# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

###############################################################################
# Import packages
###############################################################################

from datetime import datetime
import yaml
import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb

from third_party.lookahead_pytorch.optimizer import Lookahead
from datamodule.data_loader import load_datasets
from trainingmodule.snn_training_backprop import fit
from utils.option_dicts import loss_functions, optimizers, network_types

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_snn_backprop_encodinglayer.yaml")
    args = parser.parse_args()

    # Config from yaml file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["device"] == 'cuda':
        torch.cuda.empty_cache()

    NGENS = config["train"]["gens"]
    device = config["device"]
    
    # model parameters
    input_size = config["net"]["input_size"]
    hidden_size = config["net"]["hidden_size"]
    output_size = config["net"]["output_size"]

    # Load training and validation dataloaders
    train_loader, val_loader = load_datasets(config)

    # Build the spiking model
    SNN = network_types[config["type"]["net"]]
    model = SNN(input_size, 
                hidden_size, 
                output_size, 
                record=False, 
                dt=config["net"]["dt"], 
                neuron_params=config["net"]["neuron_params"], 
                learn_params=config["train"]["learn_params"])
    model = model.to(device)
    # model.share_memory()

    optimizer = optimizers[config["train"]["optimizer"]](model.parameters(), lr=config["train"]["learning_rate"])
    optimizer = Lookahead(optimizer)
    loss_function = loss_functions[config["type"]["loss_func"]]

    # losses = torch.zeros(NGENS, 10).to(device)
    # losses.share_memory_()

    # initialize weights and biases
    wandb.init(project="imusnn", entity="sstroobants", config=config)
    wandb.watch(model, log_freq=50, log="all")

    ################################################################################
    ## Create output folder and saving config to file
    ################################################################################

    # filename for model storage
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    out_dir = f'runs/snn_backprop{dt_string}'
    os.mkdir(out_dir)

    config_save_file = f'{out_dir}/config.txt'
    with open(config_save_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    ################################################################################
    ## Train model
    ################################################################################

    best_model, best_fitness, ngens, avg_time, fitness_hist, val_hist = fit(model, train_loader, val_loader, optimizer, NGENS, out_dir, device)
    print(f'Best obtained loss was : {best_fitness}')

    ################################################################################
    ## Saving results to file
    ################################################################################

    results_save_file = f'{out_dir}/results.txt'
    results = {
                "best_loss": float(best_fitness),
                "avg_gen_calc_time": avg_time,
                "number_generations": ngens, 
                "fitness_hist": fitness_hist,
                "validation_hist": val_hist
    }

    with open(results_save_file, 'w') as f:
        yaml.dump(results, f, sort_keys=False)
