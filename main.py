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

def worker(gpu, args):
    # Setting up gpu and rank within DDP
    rank = args.node_rank * args.gpus + gpu
    torch.cuda.set_device(gpu)
    device = torch.device("cuda")

    # Logger init
    logger = None
    if rank == 0 or not args.ddp:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        rfh = RotatingFileHandler(args.logging_file, maxBytes=100000, backupCount=10, encoding="UTF-8")
        logger.addHandler(rfh)
    if args.ddp:
        #dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        dist.init_process_group(backend='gloo', init_method='file://'+args.sync_path, world_size=args.world_size, rank=rank)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data init
    if not args.evaluate and not args.force_align and not args.force_align_ctc and not args.force_align_mfa:
        data = AsrDataset(args, args.train_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    else:
        data = AsrDataset(args, args.valid_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)

    # Loading models
    if not args.force_align_mfa:
        print(f'Loading model.')
        model = LAS(args)
        print(f'# model parameters = {count_parameters(model)/1e6}M')
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        normalizer = InputNormalization(update_until_epoch=3)
        if os.path.isfile(args.ckpt_path) and args.ckpt_path != '':
            print(f'Loading checkpoint ...')
            checkpoint = torch.load(args.ckpt_path, map_location=f'cuda:{gpu}')
            print(f'Done.')
            if args.load_norm:
                normalizer = checkpoint['normalizer'].to('cpu')
            if args.load_opt:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.epochs_done = checkpoint['epochs_done']
        else:
            checkpoint = None

        if len(NEW_TAGS) == 0:
            if args.ddp:
                model = nn.parallel.DistributedDataParallel(model)
            if checkpoint is not None:
                print(f'Loading parameters ...')
                load_dict(model, checkpoint['state_dict'], ddp=args.ddp)
                print(f'Done.')
        else:
            if checkpoint is not None:
                print(f'Loading parameters ...')
                load_dict(model, checkpoint['state_dict'], ddp=False)
                print(f'Done.')
            model.update_weight(OLDLEN, NEWLEN, device)
            args.vocab_size = NEWLEN
            print(f'New model size = {count_parameters(model)/1e6}M')
            if args.ddp:
                model = nn.parallel.DistributedDataParallel(model)

    # Define sampler for DDP and training utility
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data, num_replicas=args.world_size, rank=rank)
    else:
        sampler = None
    if args.force_align:
        aligner = ForcedAligner(args, data, device, normalizer)
    elif args.force_align_ctc:
        aligner = CTCForcedAligner(args, data, device, normalizer)
    elif args.force_align_mfa:
        aligner = MFA(args, data)
    else:
        trainer = Trainer(args, data, device, optimizer, normalizer, sampler=sampler, rank=rank, checkpoint=checkpoint)

    # Training starts here
    if args.evaluate:
        print(f'Starting evaluation ...')
        trainer.evaluate(model)
        print(f'Done')
    elif args.force_align or args.force_align_ctc:
        if args.fa_ensemble:
            print(f'Loading model 2.')
            model2 = Aligner(args)
            model2 = model2.to(device)
            print(f'Force aligning ...')
            aligner.runEns(model, model2, logger)
            print(f'Done')
        else:
            print(f'Force aligning ...')
            aligner.run(model, logger)
            print(f'Done')
    elif args.force_align_mfa:
        print(f'Force aligning ...')
        aligner.run(args.mfa_align_path, logger)
        print(f'Done')
    else:
        print(f'Starting training ...')
        trainer.asr(model, logger)
        print(f'Done')


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
    parser.add_argument('--mfa-align-path', type=str, default='', help='')
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
    parser.add_argument('--force-align-ctc', action='store_true', help='')
    parser.add_argument('--force-align-mfa', action='store_true', help='')
    parser.add_argument('--offset-fa', action='store_true', help='')
    parser.add_argument('--fa-ensemble', action='store_true', help='')
    parser.add_argument('--cache-gt', action='store_true', help='')
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nnodes
    args.vocab_size = OLDLEN#len(ASR_ID2TOK)
    if args.world_size > 1:
        args.ddp = True
    else:
        args.ddp = False
    if args.ddp:
        #os.environ['MASTER_ADDR'] = 'localhost'#args.address
        #os.environ['MASTER_PORT'] = '8888'
        #mp.spawn(worker, nprocs=args.gpus, args=(args,))
        local_rank = int(os.environ['LOCAL_RANK'])
        worker(local_rank, args)
    else:
        worker(args.gpu_num, args)

if __name__ == '__main__':
    main()
