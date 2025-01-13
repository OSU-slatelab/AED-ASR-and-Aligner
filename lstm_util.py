from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F

def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

# LSTM layer for pLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through pLSTM
# Note the input should have timestep%2 == 0
class pLSTMLayer(nn.Module):
    def __init__(self, inDim, hidDim, dropout=0.1):
        super(pLSTMLayer, self).__init__()
        self.pLSTM = LstmLayer(2*inDim, hidDim, dropout=dropout, bidirectional=True)
    
    def forward(self, x, lens):
        pdb.set_trace()
        x, _ = self.pLSTM(x)
        return x, lens
        
# Listener is a pLSTM stacking n layers to reduce time resolution 2^n times
class pLSTM(nn.Module):
    def __init__(self, inDim, hidDim, nLayer, dropout=0.1):
        super(pLSTM, self).__init__()
        self.nLayer = nLayer
        # Listener RNN layer
        self.layer0 = pLSTMLayer(inDim, hidDim, dropout=dropout)
        for i in range(1, nLayer):
            setattr(self, 'layer'+str(i), pLSTMLayer(2*hidDim, hidDim, dropout=dropout))

    def forward(self, x, lens):
        for i in range(self.nLayer):
            x, lens = getattr(self,'layer'+str(i))(x, lens)
        return x, lens

class CustomLSTMLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(CustomLSTMLayer, self).__init__()

        self.pLSTM = nn.LSTM(hidden_dim, hidden_dim, 1, bidirectional=False)

        self.linear1 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x, lens):
        lens = lens.cpu()
        lens = lens + input_x.size(1) - lens.max()
        # pack sequence
        pack = pack_padded_sequence(input_x, lens, batch_first=True, enforce_sorted=False)
        # forward pass - LSTM
        self.pLSTM.flatten_parameters()
        output, hidden = self.pLSTM(pack)
        # pad packed seq output of LSTM
        out_lstm_pad, _ = pad_packed_sequence(output, batch_first=True)
        # Add and norm
        out = self.norm1(input_x + self.dropout(out_lstm_pad))
        # MLP
        out2 = self.linear2(self.dropout(F.relu(self.linear1(self.dropout(out)))))
        # Add and norm
        out = self.norm2(out + self.dropout(out2))
    
        return out, lens, hidden 

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, nlayer, dropout=0.1):
        super(CustomLSTM, self).__init__()
        self.nlayer = nlayer
        self.LSTM_layer0 = CustomLSTMLayer(input_dim, dropout=dropout)
        for i in range(1,self.nlayer):
            setattr(self, 'LSTM_layer'+str(i), CustomLSTMLayer(input_dim, dropout=dropout))

    def forward(self, input, lens):
        output, lens, _  = self.LSTM_layer0(input, lens)
        for i in range(1,self.nlayer):
            output, lens, _ = getattr(self,'LSTM_layer'+str(i))(output, lens)
        return output, lens
