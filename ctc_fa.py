from util import *
from models import *
from train import *
from data import *
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
from speechbrain.processing.features import InputNormalization
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--nnodes', type=int, default=1, help='# nodes used for DDP')
    parser.add_argument('--node_rank', type=int, default=0, help='rank among nodes')
    parser.add_argument('--gpus', type=int, default=1, help='# gpus per node')
    parser.add_argument('--gpu-num', type=int, default=1, help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--conv-fac', type=int, default=3, help='')
    parser.add_argument('--nspeech-feat', type=int, default=80, help='# logmels')
    parser.add_argument('--sample-rate', type=int, default=16000, help='speech sampling rate to use')
    parser.add_argument('--batch-size', type=int, default=64, help='')
    parser.add_argument('--bsz-small', type=int, default=8, help='batch size per gpu')
    parser.add_argument('--nepochs', type=int, default=60, help='')
    parser.add_argument('--epochs-done', type=int, default=0, help='')
    parser.add_argument('--checkpoint-after', type=int, default=1, help='')
    parser.add_argument('--n-layer', type=int, default=6, help='')
    parser.add_argument('--in-dim', type=int, default=320, help='')
    parser.add_argument('--head-dim', type=int, default=64, help='')
    parser.add_argument('--nhead', type=int, default=1, help='')
    parser.add_argument('--head-fa', type=int, default=7, help='')
    parser.add_argument('--res-fa', type=int, default=120, help='')
    parser.add_argument('--roll-fac', type=int, default=1, help='')
    parser.add_argument('--lam', type=float, default=0.2, help='')
    parser.add_argument('--ctc-wt', type=float, default=0.0, help='')
    parser.add_argument('--thres-fa', type=float, default=0.04, help='')
    parser.add_argument('--attn-type', type=str, default='content', help='')
    parser.add_argument('--logging-file', type=str, default='logs/scratch.log', help='')
    parser.add_argument('--train-path', type=str, default='', help='')
    parser.add_argument('--valid-path', type=str, default='', help='')
    parser.add_argument('--ckpt-path', type=str, default='', help='')
    parser.add_argument('--save-path', type=str, default='', help='')
    parser.add_argument('--gt-path', type=str, default='', help='')
    parser.add_argument('--address', type=str, default='localhost', help='')
    parser.add_argument('--sync-path', type=str, default='/users/PAS1939/vishal/asr/sync/shared', help='')
    parser.add_argument('--corpus', type=str, default='librispeech', help='')
    parser.add_argument('--att-path', type=str, default='', help='')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=1.0, help='clip grad')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--prev', action='store_true', help='')
    parser.add_argument('--next', action='store_true', help='')
    parser.add_argument('--adapt-start', action='store_true', help='')
    parser.add_argument('--dont-fix-path', action='store_true', help='')
    parser.add_argument('--load-norm', action='store_true', help='')
    parser.add_argument('--load-opt', action='store_true', help='')
    parser.add_argument('--load-sch', action='store_true', help='')
    parser.add_argument('--evaluate', action='store_true', help='')
    parser.add_argument('--force-align', action='store_true', help='')
    parser.add_argument('--offset-fa', action='store_true', help='')
    parser.add_argument('--fa-ensemble', action='store_true', help='')
    parser.add_argument('--cache-gt', action='store_true', help='')
    
    args = parser.parse_args()
    args.vocab_size = OLDLEN#len(ASR_ID2TOK)

    torch.cuda.set_device(args.gpu_num)
    device = torch.device("cuda")
    data = AsrDataset(args, args.valid_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    print(f'Loading model.')
    model = LAS(args)
    print(f'# model parameters = {count_parameters(model)/1e6}M')
    model = model.to(device)
    model.eval()

    collator = Collator(args)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator, pin_memory=True, sampler=None)

    for speechB, textB, textOut, textCtc, logitLens, lmax, targetLens, _ in tqdm(self.loader):
        if speechB is None:
            continue
        lmax = speechB.size(1)
        lens_norm = [1.*(x/lmax) for x in logitLens]
        speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
        speechB, logitLens = roll_in(speechB, logitLens, fac=self.args.roll_fac) # lower sequence length
        speechB, textB, textOut, textCtc = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(textOut, self.device), load2gpu(textCtc, self.device)
        _, ctc_logits, _ = model(speechB, textB, logitLens, targetLens)
        
        emission = ctc_logits[0].cpu().detach()


if __name__ == '__main__':
    main()
