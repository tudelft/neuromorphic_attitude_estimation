# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import matplotlib.pyplot as plt

from norse.torch import LIFParameters, LIFState, LIFFeedForwardState, LIParameters
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch import LICell, LIState

from typing import NamedTuple

class SNNState(NamedTuple):
    li : LIState

class SNNDecode(torch.nn.Module):
    def __init__(self, hidden_features, 
                        output_features, 
                        record=False, 
                        dt=0.001, 
                        neuron_params=None, 
                        learn_params=True,
                        neuron_mask=None):
        super(SNNDecode, self).__init__()

        if neuron_params == None:
            alpha = 20
            v_thresh = 0.5
            tau_syn_out = 0.5
            tau_mem_out = 0.5
        else:
            alpha = torch.tensor(neuron_params["alpha"])
            tau_syn_out = neuron_params["tau_syn_out"]
            tau_mem_out = neuron_params["tau_mem_out"]

        self.dt = dt
        self.alpha = alpha

        if learn_params:
            self.tau_syn_out = torch.nn.Parameter(torch.as_tensor(tau_syn_out, dtype=torch.float))
            self.tau_mem_out = torch.nn.Parameter(torch.as_tensor(tau_mem_out, dtype=torch.float))
        else:
            self.tau_syn_out = torch.as_tensor(tau_syn_out, dtype=torch.float)
            self.tau_mem_out = torch.as_tensor(tau_mem_out, dtype=torch.float)

        self.fc_output = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(p=LIParameters(tau_syn_inv= self.tau_syn_out, 
                            tau_mem_inv = self.tau_mem_out),
                            dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

        self.neuron_mask = neuron_mask

        
    def forward(self, x, so=None):

        self.out.p = LIParameters(
                    tau_syn_inv = self.tau_syn_out.clamp(0, 1),
                    tau_mem_inv = self.tau_mem_out.clamp(0, 1))
        
        batch_size, seq_length, ninputs = x.size()
        x = x.permute(1, 0, 2)

        # If using 1 timestep only, this should be given as input I suppose.
        # s1 = z_rec = senc = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIState(
                    v = torch.zeros(seq_length, batch_size, self.hidden_features),
                    i = torch.zeros(seq_length, batch_size, self.hidden_features)
                )
            )

        for ts in range(seq_length):
            z = x[ts, :, :]
            if self.neuron_mask != None:
                z[:, self.neuron_mask[0]] = 0

            z = self.fc_output(z)
            vo, so = self.out(z, so)
            
            voltages += [vo]
            # voltages += [z]

            if self.record:
                self.recording.lif0.v[ts,:] = so.v
                self.recording.lif0.i[ts,:] = so.i

        output = torch.stack(voltages)

        return output, so

    def reset(self):
        pass