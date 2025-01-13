from util import *
from models import *
from train import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
from speechbrain.processing.features import InputNormalization
import torch.nn as nn
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import sys
import resource
import torch.distributed as dist
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class CollatorDec(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        speechL = [x[0].squeeze(0) for x in lst]
        pack1 = pack_sequence(speechL, enforce_sorted=False)
        speechB, logitLens = pad_packed_sequence(pack1, batch_first=True)
        lmax = speechB.size(1)

        text = [x[1] for x in lst]
        key = [x[2] for x in lst]

        return speechB, text, logitLens, lmax, key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--world-size', type=int, default=1, help='')
    parser.add_argument('--rank', type=int, default=0, help='')
    parser.add_argument('--gpu-num', type=int, default=-1, help='')
    parser.add_argument('--nspeech-feat', type=int, default=80, help='# logmels')
    parser.add_argument('--sample-rate', type=int, default=16000, help='speech sampling rate to use')
    parser.add_argument('--wav-len', type=int, default=45, help='')
    parser.add_argument('--nhead', type=int, default=1, help='')
    parser.add_argument('--beam-size', type=int, default=16, help='')
    parser.add_argument('--dropout', type=float, default=0.25, help='')
    parser.add_argument('--attn-type', type=str, default='content', help='')
    parser.add_argument('--test-path', type=str, default='', help='')
    parser.add_argument('--ckpt-path', type=str, default='', help='')
    parser.add_argument('--decode-path', type=str, default='', help='')
    parser.add_argument('--enc-type', type=str, default='lstm', help='')
    parser.add_argument('--corpus', type=str, default='librispeech', help='')
    parser.add_argument('--gt-path', type=str, default='', help='')
    parser.add_argument('--dont-fix-path', action='store_true', help='')
    parser.add_argument('--length-norm', action='store_true', help='')
    
    args = parser.parse_args()
    args.vocab_size = len(ASR_ID2TOK)
    ##
    args.rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    ##
    if args.gpu_num > 0:
        torch.cuda.set_device(args.gpu_num)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data init
    csv_path = os.path.join(args.test_path)#, f'{args.rank}.csv')
    data = AsrDataset(args, csv_path,  n_mels=args.nspeech_feat, sample_rate=args.sample_rate, train=False)
    ##
    sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=args.world_size, rank=args.rank)
    #sampler = None
    ##
    collator = CollatorDec(args)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator, sampler=sampler)

    # Load model
    print(f'Loading model.')
    model = LAS(args)
    print(f'# model parameters = {count_parameters(model)/1e6}M')
    print(f'Loading checkpoint.')
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    load_dict(model, checkpoint['state_dict'], ddp=False)
    normalizer = checkpoint['normalizer'].to('cpu')
    model.eval()

    # Decode
    if args.rank==0 and not os.path.exists(args.decode_path):
        os.makedirs(args.decode_path)
    if args.rank != 0:
        while not os.path.exists(args.decode_path):
            continue

    write_path = os.path.join(args.decode_path, f'{args.rank}.txt')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rfh = RotatingFileHandler(os.path.join(args.decode_path, f'logs{args.rank}.log'), maxBytes=1000000, backupCount=10, encoding="UTF-8")
    logger.addHandler(rfh)
    with open(write_path, 'w') as dP:
        for speechB, text, logitLens, lmax, key in tqdm(loader, disable=(args.rank!=0)):#, file=sys.stdout):
            if speechB is None:
                continue
            GT = text[0]
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            hyp, score = model.beam_search(speechB, logitLens, beam_size=args.beam_size)
            hypText = convert_id2tok(hyp)
            logger.info(f'{key[0]} ----> {GT} ----> {hypText}')
            dP.write(f'{key[0]} ----> {GT} ----> {hypText}\n')
        #dP.write(f'{key[0]} ----> {GT} ----> {hypText}\n')

if __name__ == '__main__':
    main()
