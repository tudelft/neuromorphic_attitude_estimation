# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from time import time

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs, dropout=0.1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.batch_size = batch_size

        # self.drop = nn.Dropout(dropout)
        
        # self.norm_gyro = nn.BatchNorm1d(1)
        # self.norm_acc = nn.BatchNorm1d(1)

        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, nonlinearity='relu')
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_outputs, bias=False)
        # self.activation = nn.Tanh()
        # self.init_weights()


    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size), 
        #           weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size, dtype=weight.dtype)
        return hidden

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.lstm.weight_hh_l0, -initrange, initrange)
        nn.init.uniform_(self.lstm.weight_hh_l1, -initrange, initrange)
        nn.init.uniform_(self.lstm.weight_ih_l0, -initrange, initrange)
        nn.init.uniform_(self.lstm.weight_ih_l0, -initrange, initrange)
        nn.init.uniform_(self.lstm.bias_hh_l0, -initrange, initrange)
        nn.init.uniform_(self.lstm.bias_hh_l1, -initrange, initrange)
        nn.init.uniform_(self.lstm.bias_ih_l0, -initrange, initrange)
        nn.init.uniform_(self.lstm.bias_ih_l0, -initrange, initrange)
        nn.init.zeros_(self.fc.weight)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)

    def forward(self, x, hidden):
        # out, hidden = self.rnn(x, hidden)  
        
        out, hidden = self.gru(x, hidden)
        
        # out, hidden = self.lstm(x, hidden)

        # out = self.drop(out)
        out = self.fc(out)
        # out = self.activation(out)
        # print(out)
        # out = out / torch.linalg.norm(out, dim=2, keepdim=True)

        out = out.permute([1, 0, 2])
        return out, hidden
    