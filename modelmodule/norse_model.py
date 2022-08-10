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
from time import time

class SNNState(NamedTuple):
    lif0 : LIFFeedForwardState
    lif1 : LIFState


class SNN(torch.nn.Module):
    def __init__(self, input_features, 
                        hidden_features, 
                        output_features, 
                        record=False, 
                        dt=0.001, 
                        with_delay=False, 
                        neuron_params=None, 
                        learn_params=True,
                        neuron_mask=None):
        super(SNN, self).__init__()
        self.with_delay = with_delay

        if neuron_params == None:
            alpha = 20
            tau_syn = 0.1
            tau_mem = 0.05
            v_thresh = 1
            tau_syn_out = 0.5
            tau_mem_out = 0.5
        else:
            alpha = torch.tensor(neuron_params["alpha"])
            tau_syn = neuron_params["tau_syn"]
            tau_mem = neuron_params["tau_mem"]
            v_thresh = neuron_params["v_thresh"]
            tau_syn_out = neuron_params["tau_syn_out"]
            tau_mem_out = neuron_params["tau_mem_out"]

        self.dt = dt
        self.alpha = alpha

        if learn_params:
            self.tau_syn_enc = torch.nn.Parameter(torch.full((1, hidden_features), tau_syn, dtype=torch.float))
            self.tau_mem_enc = torch.nn.Parameter(torch.full((1, hidden_features), tau_mem, dtype=torch.float))
            self.tau_syn_l1 = torch.nn.Parameter(torch.full((1, hidden_features), tau_syn, dtype=torch.float))
            self.tau_mem_l1 = torch.nn.Parameter(torch.full((1, hidden_features), tau_mem, dtype=torch.float))
            self.tau_syn_out = torch.nn.Parameter(torch.as_tensor(tau_syn_out, dtype=torch.float))
            self.tau_mem_out = torch.nn.Parameter(torch.as_tensor(tau_mem_out, dtype=torch.float))
        else:
            self.tau_syn_enc = torch.full((1, hidden_features), tau_syn, dtype=torch.float)
            self.tau_mem_enc = torch.full((1, hidden_features), tau_mem, dtype=torch.float)
            self.tau_syn_l1 = torch.full((1, hidden_features), tau_syn, dtype=torch.float)
            self.tau_mem_l1 = torch.full((1, hidden_features), tau_mem, dtype=torch.float)
            self.tau_syn_out = torch.as_tensor(tau_syn_out, dtype=torch.float)
            self.tau_mem_out = torch.as_tensor(tau_mem_out, dtype=torch.float)
        
        # self.vth_enc = torch.nn.Parameter(torch.full((1, hidden_features), v_thresh, dtype=torch.float))
        self.vth_enc = v_thresh

        # self.vth_l1 = torch.nn.Parameter(torch.full((1, hidden_features), v_thresh, dtype=torch.float))
        self.vth_l1 = v_thresh
        


        self.fc_input = torch.nn.Linear(input_features, hidden_features, bias=False)

        self.encode = LIFCell(
            p=LIFParameters(alpha=alpha, 
                            tau_syn_inv= torch.as_tensor(self.tau_syn_enc), 
                            tau_mem_inv = torch.as_tensor(self.tau_mem_enc), 
                            v_th=torch.tensor(self.vth_enc)),
            dt=dt
        )

        self.l1 = LIFRecurrentCell(
            hidden_features,
            hidden_features,
            p=LIFParameters(alpha=alpha, 
                            tau_syn_inv= torch.as_tensor(self.tau_syn_l1), 
                            tau_mem_inv = torch.as_tensor(self.tau_mem_l1), 
                            v_th=torch.tensor(self.vth_l1)),
            dt=dt, 
            autapses=True                     
        )

        self.input_features = input_features
        self.fc_output = torch.nn.Linear(hidden_features, output_features, bias=False)

        self.out = LICell(p=LIParameters(tau_syn_inv= self.tau_syn_out, 
                            tau_mem_inv = self.tau_mem_out),
                            dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

        self.neuron_mask = neuron_mask
        self.dropout = torch.nn.Dropout(p=0.05)
        

        
    def forward(self, x, s1=None, z_rec=None, senc=None, so=None):

        # if self.training:
        self.encode.p = LIFParameters(
                    tau_syn_inv = self.tau_syn_enc.clamp(0, 1),
                    tau_mem_inv = self.tau_mem_enc.clamp(0, 1), 
                    v_th=self.encode.p.v_th.clone().detach(),
                    alpha=self.alpha)
        self.l1.p = LIFParameters(
                    tau_syn_inv = self.tau_syn_l1.clamp(0, 1),
                    tau_mem_inv = self.tau_mem_l1.clamp(0, 1), 
                    v_th=self.l1.p.v_th.clone().detach(),
                    alpha=self.alpha)
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
                LIFState(
                    z = torch.zeros(seq_length, batch_size, self.hidden_features),
                    v = torch.zeros(seq_length, batch_size, self.hidden_features),
                    i = torch.zeros(seq_length, batch_size, self.hidden_features)
                ),
                LIFState(
                    z = torch.zeros(seq_length, batch_size, self.hidden_features),
                    v = torch.zeros(seq_length, batch_size, self.hidden_features),
                    i = torch.zeros(seq_length, batch_size, self.hidden_features)
                )
            )

        times = []
        for ts in range(seq_length):
            z = x[ts, :, :]
            z = self.fc_input(z)
            z, senc = self.encode(z, senc)

            if self.neuron_mask != None:
                z[:, self.neuron_mask[0]] = 0
            z_enc = z.clone()

            # z = self.dropout(z)
            # t = time()
            z_rec_next, s1 = self.l1(z, s1)
            # t_diff = time() - t
            # times.append(t_diff)

            if self.neuron_mask != None:
                z_rec_next[:, self.neuron_mask[1]] = 0

            # z_rec_next, s2 = self.l2(z_rec_next, s2)
            
            if not self.with_delay:
              z_rec = z_rec_next

            # initialize spikes to outer layer at zero (only necessary with delay) 
            if z_rec == None:
              z_rec = torch.zeros_like(z_rec_next)

            # z_rec = self.dropout(z_rec)
            z = self.fc_output(z_rec)
            vo, so = self.out(z, so)
            
            voltages += [vo]
            # voltages += [z]

            if self.record:
                self.recording.lif0.z[ts,:] = z_enc
                self.recording.lif0.v[ts,:] = senc.v
                self.recording.lif0.i[ts,:] = senc.i
                self.recording.lif1.z[ts,:] = s1.z
                self.recording.lif1.v[ts,:] = s1.v
                self.recording.lif1.i[ts,:] = s1.i

            if self.with_delay:
              z_rec = z_rec_next
        # print(np.mean(times))
        output = torch.stack(voltages)

        # print(output.size())
        return output, [s1, z_rec, senc, so]

    def reset(self):
        pass