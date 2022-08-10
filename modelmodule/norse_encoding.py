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
    lif0 : LIFState


class SNNEncode(torch.nn.Module):
    def __init__(self, input_features, 
                        hidden_features, 
                        record=False, 
                        dt=1,
                        neuron_params=None, 
                        learn_params=True,
                        neuron_mask=None):
        super(SNNEncode, self).__init__()

        if neuron_params == None:
            alpha = 20
            tau_syn = 0.1
            tau_mem = 0.05
            v_thresh = 1
        else:
            alpha = torch.tensor(neuron_params["alpha"])
            tau_syn = neuron_params["tau_syn"]
            tau_mem = neuron_params["tau_mem"]
            v_thresh = neuron_params["v_thresh"]

        self.dt = dt
        self.alpha = alpha

        if learn_params:
            self.tau_syn_enc = torch.nn.Parameter(torch.full((1, hidden_features), tau_syn, dtype=torch.float))
            self.tau_mem_enc = torch.nn.Parameter(torch.full((1, hidden_features), tau_mem, dtype=torch.float))
        else:
            self.tau_syn_enc = torch.full((1, hidden_features), tau_syn, dtype=torch.float)
            self.tau_mem_enc = torch.full((1, hidden_features), tau_mem, dtype=torch.float)
        
        self.vth_enc = v_thresh
        self.fc_input = torch.nn.Linear(input_features, hidden_features, bias=False)

        self.encode = LIFCell(
            p=LIFParameters(alpha=alpha, 
                            tau_syn_inv= torch.as_tensor(self.tau_syn_enc), 
                            tau_mem_inv = torch.as_tensor(self.tau_mem_enc), 
                            v_th=torch.tensor(self.vth_enc)),
            dt=dt
        )

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.record = record
        self.neuron_mask = neuron_mask
        

        
    def forward(self, x, senc=None):
        self.encode.p = LIFParameters(
                    tau_syn_inv = self.tau_syn_enc.clamp(0, 1),
                    tau_mem_inv = self.tau_mem_enc.clamp(0, 1), 
                    v_th=self.encode.p.v_th.clone().detach(),
                    alpha=self.alpha)
        
        batch_size, seq_length, ninputs = x.size()
        x = x.permute(1, 0, 2)

        output_spikes = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z = torch.zeros(seq_length, batch_size, self.hidden_features),
                    v = torch.zeros(seq_length, batch_size, self.hidden_features),
                    i = torch.zeros(seq_length, batch_size, self.hidden_features)
                )
            )

        for ts in range(seq_length):
            z = x[ts, :, :]
            z = self.fc_input(z)
            z, senc = self.encode(z, senc)

            if self.neuron_mask != None:
                z[:, self.neuron_mask[0]] = 0
            z_enc = z.clone()

            output_spikes += [z_enc]
            if self.record:
                self.recording.lif0.z[ts,:] = z_enc
                self.recording.lif0.v[ts,:] = senc.v
                self.recording.lif0.i[ts,:] = senc.i

        output = torch.stack(output_spikes)
        return output, senc

    def reset(self):
        pass