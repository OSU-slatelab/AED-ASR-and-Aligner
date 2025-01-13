import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.RNN import LSTMCell, GRUCell, RNNCell

class MheadContentBasedAttention(nn.Module):
    """ This class implements content-based attention module for seq2seq
    learning.

    Reference: NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN
    AND TRANSLATE, Bahdanau et.al. https://arxiv.org/pdf/1409.0473.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim, nhead, scaling=1.0):
        super(MheadContentBasedAttention, self).__init__()

        self.attn_dim = attn_dim
        self.nhead = nhead
        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, nhead, bias=False)
        self.mlp_out = nn.Linear(nhead * enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None


    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.

        """
        
        bsz = enc_states.size(0)
        if self.precomputed_enc_h is None:

            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h)
        ).permute(0,2,1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask.unsqueeze(1) == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # compute context vectors
        # [B, nhead, L] X [B, L, F]
        context = torch.bmm(attn, enc_states).view(bsz,-1)
        context = self.mlp_out(context)

        return context, attn

class MheadAttentionalRNNDecoder(nn.Module):
    def __init__(
        self,
        nhead,
        rnn_type,
        attn_type,
        hidden_size,
        attn_dim,
        num_layers,
        enc_dim,
        input_size,
        nonlinearity="relu",
        re_init=True,
        normalization="batchnorm",
        scaling=1.0,
        channels=None,
        kernel_size=None,
        bias=True,
        dropout=0.0,
    ):
        super(MheadAttentionalRNNDecoder, self).__init__()

        self.nhead = nhead
        self.rnn_type = rnn_type.lower()
        self.attn_type = attn_type.lower()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.scaling = scaling
        self.bias = bias
        self.dropout = dropout
        self.normalization = normalization
        self.re_init = re_init
        self.nonlinearity = nonlinearity

        # only for location-aware attention
        self.channels = channels
        self.kernel_size = kernel_size

        # Combining the context vector and output of rnn
        self.proj = nn.Linear(
            self.hidden_size + self.attn_dim, self.hidden_size
        )

        if self.attn_type == "content":
            self.attn = MheadContentBasedAttention(
                enc_dim=enc_dim,
                dec_dim=self.hidden_size,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
                nhead=self.nhead,
                scaling=self.scaling,
            )
        else:
            raise ValueError(f"{self.attn_type} is not implemented.")

        self.drop = nn.Dropout(p=self.dropout)

        # set dropout to 0 when only one layer
        dropout = 0 if self.num_layers == 1 else self.dropout

        # using cell implementation to reduce the usage of memory
        if self.rnn_type == "rnn":
            cell_class = RNNCell
        elif self.rnn_type == "gru":
            cell_class = GRUCell
        elif self.rnn_type == "lstm":
            cell_class = LSTMCell
        else:
            raise ValueError(f"{self.rnn_type} not implemented.")

        kwargs = {
            "input_size": input_size + self.attn_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bias": self.bias,
            "dropout": dropout,
            "re_init": self.re_init,
        }
        if self.rnn_type == "rnn":
            kwargs["nonlinearity"] = self.nonlinearity

        self.rnn = cell_class(**kwargs)

    def forward_step(self, inp, hs, c, enc_states, enc_len):
        """One step of forward pass process.

        Arguments
        ---------
        inp : torch.Tensor
            The input of current timestep.
        hs : torch.Tensor or tuple of torch.Tensor
            The cell state for RNN.
        c : torch.Tensor
            The context vector of previous timestep.
        enc_states : torch.Tensor
            The tensor generated by encoder, to be attended.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        dec_out : torch.Tensor
            The output tensor.
        hs : torch.Tensor or tuple of torch.Tensor
            The new cell state for RNN.
        c : torch.Tensor
            The context vector of the current timestep.
        w : torch.Tensor
            The weight of attention.
        """
        cell_inp = torch.cat([inp, c], dim=-1)
        cell_inp = self.drop(cell_inp)
        cell_out, hs = self.rnn(cell_inp, hs)

        c, w = self.attn(enc_states, enc_len, cell_out)
        dec_out = torch.cat([c, cell_out], dim=1)
        dec_out = self.proj(dec_out)

        return dec_out, hs, c, w

    def forward(self, inp_tensor, enc_states, wav_len):
        """This method implements the forward pass of the attentional RNN decoder.

        Arguments
        ---------
        inp_tensor : torch.Tensor
            The input tensor for each timesteps of RNN decoder.
        enc_states : torch.Tensor
            The tensor to be attended by the decoder.
        wav_len : torch.Tensor
            This variable stores the relative length of wavform.

        Returns
        -------
        outputs : torch.Tensor
            The output of the RNN decoder.
        attn : torch.Tensor
            The attention weight of each timestep.
        """
        # calculating the actual length of enc_states
        enc_len = torch.round(enc_states.shape[1] * wav_len).long()

        # initialization
        self.attn.reset()
        c = torch.zeros(
            enc_states.shape[0], self.attn_dim, device=enc_states.device
        )
        hs = None

        # store predicted tokens
        outputs_lst, attn_lst = [], []
        for t in range(inp_tensor.shape[1]):
            outputs, hs, c, w = self.forward_step(
                inp_tensor[:, t], hs, c, enc_states, enc_len
            )
            outputs_lst.append(outputs)
            attn_lst.append(w)

        # [B, L_d, hidden_size]
        outputs = torch.stack(outputs_lst, dim=1)

        # [B, L_d, L_e]
        attn = torch.stack(attn_lst, dim=2)

        return outputs, attn
