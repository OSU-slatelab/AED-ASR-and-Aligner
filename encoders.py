import pdb
import torch
import random
import torch.nn as nn
from util import *
from conformer import ConformerBlock
from speechbrain.lobes.augment import SpecAugment

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConformerLayer(nn.Module):
    def __init__(self, hidDim, headDim, nhead, dropout=0.25):
        super(ConformerLayer, self).__init__()
        self.encoder = ConformerBlock(dim = hidDim, dim_head=headDim, heads=nhead, ff_mult = 4, conv_expansion_factor = 2, conv_kernel_size = 32, attn_dropout = dropout, ff_dropout = dropout, conv_dropout = dropout)
        
    def forward(self, x):
        return self.encoder(x)

class LstmLayer(nn.Module):
    def __init__(self, inDim, hidDim, dropout=0.25, bidirectional=True):
        super(LstmLayer, self).__init__()
        self.encoder = nn.LSTM(inDim, hidDim, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.encoder(x)

    def step(self, x, hidden):
        return self.encoder(x, hidden)
 
class ConformerEncoder(nn.Module):
    def __init__(self, nLayer, hidDim, headDim, nhead, dropout=0.25, spec=False):
        super(ConformerEncoder, self).__init__()
        self.nLayer = nLayer
        for i in range(self.nLayer):
            setattr(self, 'Conf'+str(i), ConformerLayer(hidDim, headDim, nhead, dropout=dropout))
        self.spec = spec
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 100), time_mask_width=(0, 30))

    def augment(self, x):
        if not self.training:
            return x
        return self.aug(x)

    def forward(self, x):
        full = []
        regLyr = random.sample(list(range(self.nLayer)), 1)
        for i in range(self.nLayer):
            x = getattr(self, 'Conf'+str(i))(x)
            if self.spec and i in regLyr:
                x = self.augment(x)
            full.append(x)
        return x, full

class LstmEncoder(nn.Module):
    def __init__(self, nLayer, inDim, hidDim, dropout0=0.25, dropout=0.25, spec=False, bidirectional=True):
        super(LstmEncoder, self).__init__()
        self.nLayer = nLayer
        self.Lstm0 = LstmLayer(inDim, hidDim, dropout=dropout0, bidirectional=bidirectional)
        #self.lin0 = nn.Linear(2*hidDim, hidDim)
        #self.inNorm0 = nn.LayerNorm(hidDim)
        for i in range(1, self.nLayer):
            if bidirectional:
                setattr(self, 'Lstm'+str(i), LstmLayer(2*hidDim, hidDim, dropout=dropout, bidirectional=True))
            else:
                setattr(self, 'Lstm'+str(i), LstmLayer(hidDim, hidDim, dropout=dropout, bidirectional=False))
            #setattr(self, 'lin'+str(i), nn.Linear(2*hidDim, hidDim))
            #setattr(self, 'inNorm'+str(i), nn.LayerNorm(hidDim))
        self.spec = spec
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 100), time_mask_width=(0, 20))
        self.dropout = nn.Dropout(dropout)

    def augment(self, x):
        if not self.training:
            return x
        return self.aug(x)

    def forward(self, x):
        full = []
        regLyr = random.sample(list(range(self.nLayer)), 1)
        for i in range(self.nLayer):
            x, _ = getattr(self, 'Lstm'+str(i))(x)
            #xo = getattr(self, 'lin'+str(i))(self.dropout(xo))
            #x = getattr(self, 'inNorm'+str(i))(xo)
            #xo = getattr(self, 'inNorm'+str(i))(x)
            #x, _ = getattr(self, 'Lstm'+str(i))(xo)
            #x = x + self.dropout(xo)
            if self.spec and i in regLyr:
                x = self.augment(x)
            full.append(x)
        return x, full

    def step(self, x, hidden):
        return self.Lstm0.step(x, hidden)
        #hiddenO = []
        #for i in range(self.nLayer):
        #    x, hidden = getattr(self, 'Lstm'+str(i)).step(x, hiddenI[i])
        #    hiddenO.append(hidden)
        #return x, hiddenO

class pLSTMLayer(nn.Module):
    def __init__(self, inDim, hidDim, dropout=0.1, bidirectional=True):
        super(pLSTMLayer, self).__init__()
        self.pLSTM = LstmLayer(2*inDim, hidDim, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, lens):
        x, lens = roll_in(x, lens, fac=2, device=x.device)
        x, _ = self.pLSTM(x)
        return x, lens
        
class pLSTM(nn.Module):
    def __init__(self, inDim, hidDim, nLayer, dropout=0.1, bidirectional=True):
        super(pLSTM, self).__init__()
        fac = 2 if bidirectional else 1
        self.nLayer = nLayer
        self.layer0 = pLSTMLayer(inDim, hidDim, dropout=dropout, bidirectional=bidirectional)
        for i in range(1, nLayer):
            setattr(self, 'layer'+str(i), pLSTMLayer(fac*hidDim, hidDim, dropout=dropout, bidirectional=bidirectional))

    def forward(self, x, lens):
        for i in range(self.nLayer):
            x, lens = getattr(self,'layer'+str(i))(x, lens)
        return x, lens
