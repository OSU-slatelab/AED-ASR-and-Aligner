import pdb
import time
import numpy as np
import random
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from transformers import BertForMaskedLM, BertTokenizer
from transformers import logging
from tqdm import tqdm
from util import *
from encoders import *
#from decode_util import *
from speechbrain.decoders.seq2seq import *
from speechbrain.nnet.RNN import AttentionalRNNDecoder
from attention_mh import MheadAttentionalRNNDecoder

logging.set_verbosity_error()
NEG = -10000000
HIDD = 768
LOWL = 2
HIGHL = 13
GAP = 2
TOK_NC = BertTokenizer.from_pretrained("bert-base-uncased")

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def logsumexp(a, b):
    return np.log(np.exp(a) + np.exp(b))

def get_mask(lens, device):
    #return (torch.arange(max(lens), device=device).expand(len(lens), max(lens)) >= torch.tensor(lens, device=device).unsqueeze(1)).float()
    mask = torch.ones(len(lens), max(lens), device=device)
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

def extract(tens, out_lens, trim=False):
    #out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        if not trim:
            out.append(ten[:out_lens[i]])
        else:
            out.append(ten[:out_lens[i]][1:-1])
    return torch.cat(out, dim=0)

class Attention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask, average_attn_weights=False)
        ## Add and norm
        src = Q + self.dropout(src)
        src = self.norm1(src)
        ## MLP
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ## Add and norm
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn

class BERTNC(nn.Module):
    def __init__(self):
        super(BERTNC, self).__init__()
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).bert.embeddings
        
    def forward(self, inputs):
        return self.encoder(inputs.input_ids).permute(1,0,2), 1. - inputs.attention_mask.float()

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.encoder = model.bert
        
    def forward(self, inputs):
        output = self.encoder(**inputs)
        return output.last_hidden_state.permute(1,0,2), 1. - inputs.attention_mask

    def forward_full(self, inputs):
        output = self.encoder(**inputs)
        return output.hidden_states, 1. - inputs.attention_mask

class LAS(nn.Module):
    def __init__(self, args):
        super(LAS, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        #Listen
        self.sEncoder0 = LstmLayer(args.nspeech_feat, 256, dropout=args.dropout)
        self.pyrLstm = pLSTM(512, 256, 2, dropout=args.dropout)
        self.sEncoder1 = LstmEncoder(3, 512, 256, dropout0=args.dropout, dropout=args.dropout)

        #Attend and Spell
        self.embedding = nn.Embedding(args.vocab_size, 10)
        if args.nhead > 1:
            self.decoder = MheadAttentionalRNNDecoder(nhead=args.nhead, rnn_type="lstm", attn_type=args.attn_type, hidden_size=512, attn_dim=512, num_layers=2, enc_dim=512, input_size=10, normalization="layernorm", dropout=0.01)#, channels=256, kernel_size=100) 
        else:
            self.decoder = AttentionalRNNDecoder(rnn_type="lstm", attn_type=args.attn_type, hidden_size=512, attn_dim=512, num_layers=2, enc_dim=512, input_size=10, normalization="layernorm", dropout=0.01, channels=256, kernel_size=100) 
        self.classifier = nn.Linear(512, args.vocab_size)

    def update_weight(self, old, new, device):
        if old - new == 0:
            return
        embedding_ = nn.Embedding(new, 10).to(device)
        embedding_.weight.data[:old,:] = self.embedding.weight.data
        self.embedding = embedding_

        classifier_ = nn.Linear(512, new).to(device)
        classifier_.weight.data[:old, :] = self.classifier.weight.data
        classifier_.bias.data[:old] = self.classifier.bias.data
        self.classifier = classifier_
        return

    def forward(self, speechB, textB, lensS, lensT, getW=False):
        speechB, _ = self.sEncoder0(speechB)
        speechOut0, lensS = self.pyrLstm(speechB, lensS)
        speechOut, _ = self.sEncoder1(speechOut0)
        return None, torch.log_softmax(self.classifier(self.dropout(speechOut)), dim=-1).permute(1,0,2), lensS

        wav_len = torch.tensor(lensS, device=speechB.get_device())
        wav_len = wav_len / wav_len.max()

        textBE = self.embedding(textB)
        nextFeat, attn = self.decoder(textBE, speechOut, wav_len)
        nextFeat = extract(nextFeat, lensT)
        if getW:
            return attn
        return self.classifier(self.dropout(nextFeat)), None, lensS

    def beam_search(self, speechB, lensS, beam_size=16):
        self.eval()
        #For decoding
        self.decoder.attn.reset()
        speechB, _ = self.sEncoder0(speechB)
        speechOut0, lensS = self.pyrLstm(speechB, lensS)
        speechOut, _ = self.sEncoder1(speechOut0)

        U_max = speechOut.size(1)
        hs = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        ctx = torch.zeros(1, 512)
        SOS, EOS = ASR_TOK2ID['<sos>'], ASR_TOK2ID['<eos>']
        beam = [((SOS,), 0, hs, ctx)] #(hyp, score, state, context)
        finH = {}
        cache = {}

        for i in range(U_max):
            hypL, stateL, contextL, encL = [], [], [], []
            for hyp, score, state, context in beam:
                if hyp[-1] == EOS:
                    finH[hyp[1:-1]] = score
                    continue
                hypL.append(hyp)
                stateL.append(state)
                contextL.append(context)
                encL.append(speechOut)
            if len(hypL) > 0:
                prev_labels = torch.LongTensor([hyp[-1] for hyp in hypL]) # (B,)
                prev_c = torch.cat(contextL, dim=0)
                prev_state = (torch.cat([s[0] for s in stateL], dim=1), torch.cat([s[1] for s in stateL], dim=1))
                encFull = torch.cat(encL, dim=0)
                wav_len = torch.tensor(lensS*len(hypL))
                y = self.embedding(prev_labels) # (B,10)
                dec_out, new_state, new_c, _ = self.decoder.forward_step(y, prev_state, prev_c, encFull, wav_len)
                batch_logprobs = torch.log_softmax(self.classifier(dec_out), dim=-1)
                for k, hyp in enumerate(hypL):
                    cache[hyp] = (new_state[0][:,k:k+1,:],new_state[1][:,k:k+1,:]), new_c[k:k+1]

            new_beam, k = [], 0
            for hyp, score, _, _ in beam:
                if hyp[-1] == EOS:
                    finH[hyp[1:-1]] = score
                    continue
                state, context = cache.get(hyp)
                logprobs = batch_logprobs[k]
                k+=1
                if self.args.length_norm:
                    scores = score + logprobs + i + 1#(score + logprobs) / (i+1) 
                else:
                    scores = score + logprobs
                symL = range(len(ASR_ID2TOK))
                new_hypL = [(hyp+(v,),s,state,context) for s,v in zip(scores.tolist(),symL)]
                new_beam += new_hypL

            if len(new_beam) == 0:
                break
            if len(new_beam) > beam_size:
                scores = [b[1] for b in new_beam]
                scores_np = np.array(scores)
                #faster than sorting#
                pivot = np.partition(scores_np, len(scores)-beam_size)[len(scores)-beam_size]
                beam = list(filter(lambda x:x[1]>=pivot, new_beam))
            else:
                beam = new_beam

        finH = sorted(finH.items(), key=lambda x: x[1], reverse=True)
        hyps = [h for h, _ in finH]
        if len(finH) > 0:
            hyp, score = finH[0]
        else:
            hyp, score = [], 0

        return hyp, score
