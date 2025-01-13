from models import *
from util import *
from data import *
from tqdm import tqdm
from dataclasses import dataclass
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.augment import SpecAugment
from torchaudio.functional import rnnt_loss
import torch.distributed as dist
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

class ContrastiveLoss(nn.Module):
    def __init__(self, device, temp=0.07):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.temp = temp 

    def forward(self, r1, r2): # bsz, 768
        r1 = F.normalize(r1, dim=1)
        r2 = F.normalize(r2, dim=1)
        assert r1.size(0) == r2.size(0)
        tgt = torch.eye(r1.size(0)).to(self.device)

        align = torch.matmul(r1, r2.t()) / self.temp
        al_0 = torch.log_softmax(align, dim=0)
        al_1 = torch.log_softmax(align, dim=1)

        loss_0 = -1. * self.temp * (al_0 * tgt).sum(dim=0).mean()
        loss_1 = -1. * self.temp * (al_1 * tgt).sum(dim=1).mean()
        loss = 0.5 * (loss_0 + loss_1)
        return loss

def score_jf(y_pred, y_true, lc=3):
    jaccard = 0
    macro_f1 = 0
    pres = 0
    rec = 0
    
    for i in range(len(y_pred)):
        sT = float(y_true[i][1])
        eT = float(y_true[i][2])

        sP = float(y_pred[i][1])
        eP = float(y_pred[i][2])
        
        inter = max(min(eT, eP) - max(sT, sP), 0)
        union = eT - sT + eP - sP - inter

        try:
            precision = inter / (eP - sP)
        except ZeroDivisionError:
            precision = 0.
        try:
            recall = inter / (eT - sT)
        except ZeroDivisionError:
            recall = 0.

        pres+=precision
        rec+=recall
        try:
            jaccard += 1. * (inter / union)
        except ZeroDivisionError:
            jaccard = 0.
        try:
            macro_f1 += 2. * precision * recall / (precision + recall)
        except ZeroDivisionError:
            macro_f1 += 0

    return jaccard / len(y_pred), macro_f1 / len(y_pred), pres / len(y_pred), rec / len(y_pred) 

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

class CTCForcedAligner(object):
    def __init__(self, args, data, device, normalizer):
        self.args = args
        self.normalizer = normalizer
        self.data = data
        self.device = device
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=False, num_workers=4, collate_fn=collator, pin_memory=True)

    def gt2yt(self, R, S, res=40, la=0.4):
        y = [False]*S
        ranges = [(0, 0.055)]
        for i in range(1,len(y)):
            begin = ranges[-1][1]
            end = begin + res / 1000.
            ranges.append((begin, end))
        for i, value in enumerate(y):
            range_start, range_end = ranges[i]
            if (range_start >= R[0] and range_start <= R[1]) or (range_end >= R[0] and range_end <= R[1]) or (range_start <= R[0] and range_end >= R[1]):
                y[i] = True
        return y

    def pruneSE(self, sL, eL, w_s=0.5):
        sN = [sL[0]]
        eN = [eL[0]]
        for i in range(1, len(sL)):
            if sL[i] < eL[i-1]:
                nt = w_s*sL[i]+(1-w_s)*eL[i-1]#(sL[i] + eL[i-1]) / 2
                sN.append(nt)
                eN[-1] = nt
                eN.append(eL[i])
            else:
                sN.append(sL[i])
                eN.append(eL[i])
        return sN, eN

    def get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t+1, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t+1, tokens[1:]],
            )
        return trellis

    def backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            #assert t > 0
            if t <= 0:
                return

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]

    def merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments

    def run(self, model, logger):
        model.eval()
        scoresJ, scoresF, scoresP, scoresR = [], [], [], []
        y_pred, y_true = [], []
        for textL, speechB, textB, textOut, logitLens, lmax, targetLens, keyL, mergeL, gtKeyL, gtTupL in tqdm(self.loader):
            if speechB is None:
                continue
            if not gtKeyL[0] and not self.args.cache_gt:
                continue
            if os.path.isfile(os.path.join('/research/nfs_fosler_1/vishal/alignments', self.args.corpus, keyL[0]+'.npz')) and self.args.cache_gt:
                continue
            mLst = mergeL[0]
            if not self.args.cache_gt:
                gtKey, gtTup = tuple(gtKeyL[0]), gtTupL[0]
                gtTup = [tuple(x) for x in gtTup]
            charID, seg = convert_tok2id(textL[0])
            sent = [ASR_ID2TOK[x] for x in charID]

            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, logitLens = roll_in(speechB, logitLens, fac=self.args.roll_fac) # lower sequence length
            speechB, textB, textOut = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(textOut, self.device)
            #######
            #tokens = [ASR_TOK2ID['<space>']]+textOut.tolist()[:-1]+[ASR_TOK2ID['<space>']]
            #tokens = textOut.tolist()[:-1]
            tokens = textOut.tolist()[:-1]
            transcript = ''.join([ASR_ID2TOK[x] for x in tokens]).replace('<space>','|')
            with torch.no_grad():
                _, ctc_logits, _ = model(speechB, textB, logitLens, targetLens)
            emission = ctc_logits.permute(1,0,2)[0].cpu().detach()
            trellis = self.get_trellis(emission, tokens)
            path = self.backtrack(trellis, emission, tokens)
            if not path:
                print(f'skipping {keyL[0]}')
                continue
            segments = self.merge_repeats(path, transcript)
            segments_ = segments[1:-1]
            #segments_ = segments
            sep_pos = [i for i, char in enumerate(list(transcript)[1:-1]) if char == '|']
            #sep_pos = [i for i, char in enumerate(list(transcript)) if char == '|']
            words = transcript[1:-1].split('|')
            #words = transcript.split('|')
            assert len(words) - 1 == len(sep_pos)
            start = 0.04*(segments_[0].start - 1) + 0.015
            sL, eL = [], []
            for i, pos in enumerate(sep_pos):
                end = 0.04*(segments_[pos].end - 1) + 0.015
                sL.append(start)
                eL.append(end)
                start = end
            sL.append(start)
            eL.append(0.04*(segments_[-1].end - 1) + 0.015)
            assert len(sL) == len(eL) == len(words)
            pred = [(w,s,e) for w, s, e in zip(words, sL, eL)]
            if len(pred) != len(gtTup):
                print(f'skipping {keyL[0]}')
                continue
            #######
            for i, (word, be, en) in enumerate(gtTup):
                S = emission.size(0)
                y_pred.extend(self.gt2yt((pred[i][1], pred[i][2]), S, res=self.args.res_fa, la=0.4))
                y_true.extend(self.gt2yt((be, en), S, res=self.args.res_fa, la=0.4))

            J, F, P, R = score_jf(pred, gtTup)
            scoresJ.append(J)
            scoresF.append(F)
            scoresP.append(P)
            scoresR.append(R)
        if not self.args.cache_gt:
            f1 = f1_score(y_true, y_pred, average='binary')
            log = f'| thres = {self.args.thres_fa} | precision = {np.mean(scoresP)} | recall = {np.mean(scoresR)} | fscore = {np.mean(scoresF)} | jaccard = {np.mean(scoresJ)} | f1_discrete = {f1} |'
            logger.info(log)
            print(log)

class MFA(object):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=False, num_workers=4, collate_fn=collator, pin_memory=True)

    def gt2yt(self, R, S, res=40, la=0.4):
        y = [False]*S
        ranges = [(0, 0.055)]
        for i in range(1,len(y)):
            begin = ranges[-1][1]
            end = begin + res / 1000.
            ranges.append((begin, end))
        for i, value in enumerate(y):
            range_start, range_end = ranges[i]
            if (range_start >= R[0] and range_start <= R[1]) or (range_end >= R[0] and range_end <= R[1]) or (range_start <= R[0] and range_end >= R[1]):
                y[i] = True
        return y

    def get_ground_truth(self, pred, sLen, key):
        path = os.path.join('/research/nfs_fosler_1/vishal/alignments/mfa/char', self.args.corpus, key+'.npz')
        row_labels = []
        for i in range(len(pred)):
            ###
            word = pred[i][0] #
            if i != len(pred)-1:
                word = word+' '
            row_label = np.array(self.gt2yt((pred[i][1], pred[i][2]), sLen, res=self.args.res_fa, la=0.4))
            for ch in word:
                row_labels.append(row_label)
            ###
            #row_label = np.array(self.gt2yt((pred[i][1], pred[i][2]), sLen, res=self.args.res_fa, la=0.4))
            #row_labels.append(row_label)
        last = np.where(row_label == True)[0][-1] + 1
        row_label = np.array([False]*last+[True]*(sLen-last))
        row_labels.append(row_label)
        row_labels = np.vstack(row_labels)
        col_labels = row_labels.T
        #v = np.zeros((3,col_labels.shape[1]))
        #vB = v.astype(bool)
        hS = np.array([[1] + [0]*(col_labels.shape[0]-1)]).T.astype(bool)
        #hE = np.array([[0]*(col_labels.shape[0]-1) + [1]]).T.astype(bool)

        col_labels = np.hstack([hS, col_labels])#, hE])

        Frows = np.where(np.all(col_labels == False, axis=1) == True)[0].tolist()
        for rnum in Frows:
            col_labels[rnum] = col_labels[rnum-1]
        hard_label = col_labels.astype(float)
        hard_label = hard_label / hard_label.sum(axis=1, keepdims=True)
        soft_label = hard_label
        with open(path, 'wb') as f:
            np.savez(f, hard=hard_label, soft=soft_label)

    #def get_ground_truth(self, attn, thres, key):
    #    path = os.path.join('/research/nfs_fosler_1/vishal/alignments', self.args.corpus, key+'.npz')
    #    ##
    #    #trim attn here
    #    attn = attn[:,3:] / attn[:,3:].sum(dim=1, keepdim=True)
    #    attnT = attn.T
    #    attnT = attnT / attnT.sum(dim=1, keepdim=True)
    #    ##
    #    row_labels = []
    #    for row in attn:
    #        index = self.getBlock(row, thres=thres)
    #        row_label = np.array([False]*len(row))
    #        row_label[index] = True
    #        row_labels.append(row_label)
    #    row_labels = np.vstack(row_labels)
    #    col_labels = row_labels.T
    #    v = np.zeros((3,col_labels.shape[1]))
    #    vB = v.astype(bool)
    #    h = np.array([[1]*3 + [0]*col_labels.shape[0]]).T
    #    hB = h.astype(bool)
    #    col_labels = np.hstack([hB, np.vstack([vB,col_labels])])
    #    attnT = np.hstack([h, np.vstack([v,attnT])])

    #    Frows = np.where(np.all(col_labels == False, axis=1) == True)[0].tolist()
    #    for rnum in Frows:
    #        col_labels[rnum] = col_labels[rnum-1]
    #        attnT[rnum] = attnT[rnum-1]
    #    attnT =  attnT / attnT.sum(axis=1, keepdims=True)
    #    hard_label = col_labels.astype(float)
    #    hard_label = hard_label / hard_label.sum(axis=1, keepdims=True)
    #    soft_label = attnT.astype(float)
    #    with open(path, 'wb') as f:
    #        np.savez(f, hard=hard_label, soft=soft_label)

    def run(self, align_path, logger):
        model = json.loads(open(align_path).read().strip())
        scoresJ, scoresF, scoresP, scoresR = [], [], [], []
        y_pred, y_true = [], []
        for textL, speechB, textB, textOut, logitLens, lmax, targetLens, keyL, mergeL, gtKeyL, gtTupL in tqdm(self.loader):
            if speechB is None:
                continue
            if not gtKeyL[0] and not self.args.cache_gt:
                continue
            if os.path.isfile(os.path.join('/research/nfs_fosler_1/vishal/alignments/mfa/char', self.args.corpus, keyL[0]+'.npz')) and self.args.cache_gt:
                continue
            mLst = mergeL[0]
            if not self.args.cache_gt:
                gtKey, gtTup = tuple(gtKeyL[0]), gtTupL[0]
                gtTup = [tuple(x) for x in gtTup]

            lookup = keyL[0][4:] if self.args.corpus == 'timit' else keyL[0]
            if lookup in model:
                pred = [(c,a,b) for a,b,c in model[lookup]]
            else:
                print(f'skipping {keyL[0]}')
                continue
                #duration = gtTup[-1][-1]
                #num_tok = len(gtTup)
                #s = 0
                #pred = []
                #for i in range(num_tok):
                #    e = s + num_tok / duration
                #    pred.append((gtTup[i][0], s, e))
                #    s = e
            if self.args.cache_gt:
                sLen = math.ceil(speechB.shape[1] / 4)
                self.get_ground_truth(pred, sLen, keyL[0])
            else:
                for i, (word, be, en) in enumerate(gtTup):
                    S = math.ceil(speechB.shape[1] / 4)
                    try:
                        y_pred.extend(self.gt2yt((pred[i][1], pred[i][2]), S, res=self.args.res_fa, la=0.4))
                    except:
                        print(f'skipping {keyL[0]}')
                        continue
                    y_true.extend(self.gt2yt((be, en), S, res=self.args.res_fa, la=0.4))

                J, F, P, R = score_jf(pred, gtTup)
                scoresJ.append(J)
                scoresF.append(F)
                scoresP.append(P)
                scoresR.append(R)
        if not self.args.cache_gt:
            f1 = f1_score(y_true, y_pred, average='binary')
            log = f'| thres = {self.args.thres_fa} | precision = {np.mean(scoresP)} | recall = {np.mean(scoresR)} | fscore = {np.mean(scoresF)} | jaccard = {np.mean(scoresJ)} | f1_discrete = {f1} |'
            logger.info(log)
            print(log)

class ForcedAligner(object):
    def __init__(self, args, data, device, normalizer):
        self.args = args
        self.normalizer = normalizer
        self.data = data
        self.device = device
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=False, num_workers=8, collate_fn=collator, pin_memory=True)

    def get_ground_truth(self, attn, thres, key):
        path = os.path.join('/research/nfs_fosler_1/vishal/alignments', self.args.corpus, key+'.npz')
        ##
        #trim attn here
        attn = attn[:,3:] / attn[:,3:].sum(dim=1, keepdim=True)
        attnT = attn.T
        attnT = attnT / attnT.sum(dim=1, keepdim=True)
        ##
        row_labels = []
        for row in attn:
            index = self.getBlock(row, thres=thres)
            row_label = np.array([False]*len(row))
            row_label[index] = True
            row_labels.append(row_label)
        row_labels = np.vstack(row_labels)
        col_labels = row_labels.T
        v = np.zeros((3,col_labels.shape[1]))
        vB = v.astype(bool)
        h = np.array([[1]*3 + [0]*col_labels.shape[0]]).T
        hB = h.astype(bool)
        col_labels = np.hstack([hB, np.vstack([vB,col_labels])])
        attnT = np.hstack([h, np.vstack([v,attnT])])

        Frows = np.where(np.all(col_labels == False, axis=1) == True)[0].tolist()
        for rnum in Frows:
            col_labels[rnum] = col_labels[rnum-1]
            attnT[rnum] = attnT[rnum-1]
        attnT =  attnT / attnT.sum(axis=1, keepdims=True)
        hard_label = col_labels.astype(float)
        hard_label = hard_label / hard_label.sum(axis=1, keepdims=True)
        soft_label = attnT.astype(float)
        with open(path, 'wb') as f:
            np.savez(f, hard=hard_label, soft=soft_label)

    def getBlock(self, row, thres=0.04):
        idxs = []
        i = 0
        while not idxs:
            idxs = torch.argwhere(row > thres).squeeze(1).tolist()
            thres -= 0.005
            i += 1
        scores = row[idxs].tolist()
        x = [idxs[0]]
        sc = scores[0]
        bestS = -1
        for i in range(1, len(idxs)):
            diff = idxs[i] - idxs[i-1]
            if diff == 1:
                x.append(idxs[i])
                sc += scores[i]
            else:
                if sc > bestS:
                    bestI = x
                    bestS = sc
                x = [idxs[i]]
                sc = scores[i]
        if sc > bestS:
            bestI = x
            bestS = sc
        return bestI

    def wp2w(self, lst, sL, eL):
        lstN = []
        sN = []
        eN = []
        idxs = []
        idx = [0]
        prev = lst[0]
        for i in range(1, len(lst)):
            if '#' in lst[i]:
                w = ''.join([x for x in lst[i] if x != '#'])
                prev = prev + w
                idx.append(i)
            else:
                lstN.append(prev)
                idxs.append(idx)
                prev = lst[i]
                idx = [i]
        lstN.append(prev)
        idxs.append(idx)
        
        for l in idxs:
            sN.append(sL[l[0]])
            eN.append(eL[l[-1]])
        return lstN, sN, eN
        old = list(zip(lst, sL, eL))
        new = list(zip(lstN, sN, eN))
        lstN_, sN_, eN_ = [], [], []
        for i, tok in enumerate(lstN):
            if i+1<len(lstN) and tok == "'":
                lstN_[-1] = lstN_[-1] + "'" + lstN[i+1]
                eN_[-1] = eN[i+1]
            elif i-1>=0 and lstN[i-1] == "'":
                continue
            else:
                lstN_.append(tok)
                sN_.append(sN[i])
                eN_.append(eN[i])
        new_ = list(zip(lstN_, sN_, eN_))
        return lstN_, sN_, eN_

    def ensInterval(self, sL1, sL2, eL1, eL2, w=0.5):
        sN, eN = [], []
        for s1, s2 in zip(sL1, sL2):
            sN.append(w*s1 + (1-w)*s2)
        for e1, e2 in zip(eL1, eL2):
            eN.append(w*e1 + (1-w)*e2)
        return sN, eN

    def pruneSE(self, sL, eL, w_s=0.5):
        sN = [sL[0]]
        eN = [eL[0]]
        for i in range(1, len(sL)):
            if sL[i] < eL[i-1]:
                nt = w_s*sL[i]+(1-w_s)*eL[i-1]#(sL[i] + eL[i-1]) / 2
                sN.append(nt)
                eN[-1] = nt
                eN.append(eL[i])
            else:
                sN.append(sL[i])
                eN.append(eL[i])
        return sN, eN

    def getTimes(self, sent, tens, res=120, thres=0.04, offset=False, st=0.0):
        sL = []
        eL = []
        for i in range(len(sent)):
            lst = self.getBlock(tens[i], thres=thres)
            start = (lst[0] * res) / 1000 + st#0.04#0.24#0.12
            end = ((lst[-1]+1) * res) / 1000 + st#0.04#0.24#0.12
            if offset:
                if start > 0:
                    start = start + 0.015
                end = end + 0.015
            sL.append(start)
            eL.append(end)
        sent, sL, eL = self.wp2w(sent, sL, eL)
        return tuple(sent), list(zip(sent, sL, eL)), sL, eL

    def merge(self, sent, arr, lst):
        words = []
        arrN = []
        for i, (st, en) in enumerate(lst):
            arrN.append(arr[st:en,:].mean(dim=0, keepdim=True))
            words.append(''.join(sent[st:en]))
        return torch.cat(arrN), words

    def gt2yt(self, R, S, res=40, la=0.4):
        y = [False]*S
        ranges = [(0, 0.055)]
        for i in range(1,len(y)):
            begin = ranges[-1][1]
            end = begin + res / 1000.
            ranges.append((begin, end))
        for i, value in enumerate(y):
            range_start, range_end = ranges[i]
            if (range_start >= R[0] and range_start <= R[1]) or (range_end >= R[0] and range_end <= R[1]) or (range_start <= R[0] and range_end >= R[1]):
                y[i] = True
        return y

    def run(self, model, logger):
        model.eval()
        scoresJ, scoresF, scoresP, scoresR = [], [], [], []
        y_pred, y_true = [], []
        pandasDct = {'key':[], 'wordT':[], 'alignL':[]}
        for textL, speechB, textB, textOut, logitLens, lmax, targetLens, keyL, mergeL, gtKeyL, gtTupL in tqdm(self.loader):
            if speechB is None:
                continue
            if not gtKeyL[0] and not self.args.cache_gt and self.args.corpus != 'readr':
                continue
            if os.path.isfile(os.path.join('/research/nfs_fosler_1/vishal/alignments', self.args.corpus, keyL[0]+'.npz')) and self.args.cache_gt:
                continue
            mLst = mergeL[0]
            if not self.args.cache_gt and self.args.corpus != 'readr':
                gtKey, gtTup = tuple(gtKeyL[0]), gtTupL[0]
                gtTup = [tuple(x) for x in gtTup]
            charID, seg = convert_tok2id(textL[0])
            sent = [ASR_ID2TOK[x] for x in charID]

            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, logitLens = roll_in(speechB, logitLens, fac=self.args.roll_fac) # lower sequence length
            speechB, textB, textOut = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(textOut, self.device)

            with torch.no_grad():
                attn = model(speechB, textB, logitLens, targetLens, getW=True)
            attn_trim = attn[0]#[2]
            attn = attn_trim.cpu()
            if self.args.cache_gt:
                self.get_ground_truth(attn, self.args.thres_fa, keyL[0])
            else:
                ###
                #attn = attn[:,5:-4] / attn[:,5:-4].sum(dim=1, keepdim=True) # 3: 6: 1:
                #blockS = self.getBlock(attn[0], thres=self.args.thres_fa)
                #blockE = self.getBlock(attn[-2], thres=self.args.thres_fa)
                #start, end = blockS[0], blockE[-1]
                #if start >= end:
                #    pdb.set_trace()
                #attn = attn[:,start:end+1] / attn[:,start:end+1].sum(dim=1, keepdim=True) # 3: 6: 1:
                #start = 0#start+1
                ###
                attn = attn[:,3:] / attn[:,3:].sum(dim=1, keepdim=True)
                start = 3
                attn, sent = self.merge(sent, attn, mLst)
                sign, pred, sN, eN = self.getTimes(sent, attn, res=self.args.res_fa, thres=self.args.thres_fa, offset=self.args.offset_fa, st=start*self.args.res_fa/1000)
                if self.args.corpus != 'readr' and sign != gtKey:
                    continue

                sN, eN = self.pruneSE(sN, eN, w_s=.9)
                pred = list(zip(list(sign), sN, eN))
                if self.args.corpus == 'readr':
                    wordT = []
                    alignL = []
                    key = keyL[0]
                    for word, startT, endT in pred:
                        if endT <= startT:
                            alignL.append((word, startT, startT+(2*self.args.res_fa/1000)))
                        else:
                            alignL.append((word, startT, endT))
                        wordT.append(word)
                    pandasDct['key'].append(key)
                    pandasDct['alignL'].append(alignL)
                    pandasDct['wordT'].append(tuple(wordT))
                    continue
                for i, (word, be, en) in enumerate(gtTup):
                    S = attn.size(1)
                    y_pred.extend(self.gt2yt((pred[i][1], pred[i][2]), S, res=self.args.res_fa, la=0.4))
                    y_true.extend(self.gt2yt((be, en), S, res=self.args.res_fa, la=0.4))

                J, F, P, R = score_jf(pred, gtTup)
                scoresJ.append(J)
                scoresF.append(F)
                scoresP.append(P)
                scoresR.append(R)
        if self.args.corpus == 'readr' and not self.args.cache_gt:
            pd.DataFrame(pandasDct).to_csv('/research/nfs_fosler_1/vishal/text/readr/test1_align.csv', index=False)
        elif not self.args.cache_gt:
            f1 = f1_score(y_true, y_pred, average='binary')
            log = f'| thres = {self.args.thres_fa} | precision = {np.mean(scoresP)} | recall = {np.mean(scoresR)} | fscore = {np.mean(scoresF)} | jaccard = {np.mean(scoresJ)} | f1_discrete = {f1} |'
            logger.info(log)
            print(log)

    def runEns(self, model1, model2, logger):
        print('Loading checkpoints ...')
        checkpoint1 = torch.load('/research/nfs_fosler_1/vishal/saved_models/libri960_align_contrast_prev_next_lam1.0.pth.tar', map_location=f'cuda:{self.args.gpu_num}')
        checkpoint2 = torch.load('/research/nfs_fosler_1/vishal/saved_models/libri960_align_contrast_prev_next_lam0.0.pth.tar', map_location=f'cuda:{self.args.gpu_num}')
        load_dict(model1, checkpoint1['state_dict'], ddp=False)
        load_dict(model2, checkpoint2['state_dict'], ddp=False)
        print('Done')
        model1.eval()
        model2.eval()
        scoresJ, scoresF, scoresP, scoresR = [], [], [], []
        for speechB, bertB, nextB, prevB, logitLens, keyL, gtKeyL, gtTupL in tqdm(self.loader):
            if speechB is None:
                continue
            if not gtKeyL[0]:
                continue
            gtKey, gtTup = tuple(gtKeyL[0]), gtTupL[0]
            gtTup = [tuple(x) for x in gtTup]
            sent = TOK.convert_ids_to_tokens(bertB.input_ids[0].cpu().tolist())[1:-1]
            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, logitLens = roll_in(speechB, logitLens, fac=4) # lower sequence length
            speechB, bertB = load2gpu(speechB, self.device), load2gpu(bertB, self.device)
            with torch.no_grad():
                attn1 = model1(speechB, bertB, logitLens, getW=True)
                attn2 = model2(speechB, bertB, logitLens, getW=True)
            attn_trim1 = attn1[0][:,1:-1,:]
            attn_trim2 = attn2[0][:,1:-1,:]
            attn1 = attn_trim1.cpu()[0]
            attn2 = attn_trim2.cpu()[-1]
            attn = 0.7 * attn1 + 0.3 * attn2
            #sign, pred1, sL1, eL1 = self.getTimes(sent, attn1, res=self.args.res_fa, thres=0.03, offset=self.args.offset_fa)
            #sign, pred2, sL2, eL2 = self.getTimes(sent, attn2, res=self.args.res_fa, thres=0.04, offset=self.args.offset_fa)
            #sN, eN = self.ensInterval(sL1, sL2, eL1, eL2, w=0.4)
            sign, pred, _, _ = self.getTimes(sent, attn, res=self.args.res_fa, thres=self.args.thres_fa, offset=self.args.offset_fa)
            if sign != gtKey:
                continue
            sN, eN = self.pruneSE(sN, eN)
            pred = list(zip(list(sign), sN, eN))
            J, F, P, R = score_jf(pred, gtTup)
            scoresJ.append(J)
            scoresF.append(F)
            scoresP.append(P)
            scoresR.append(R)
        log = f'| precision = {np.mean(scoresP)} | recall = {np.mean(scoresR)} | fscore = {np.mean(scoresF)} | jaccard = {np.mean(scoresJ)} |'
        logger.info(log)
        print(log)

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer, sampler=None, rank=0, checkpoint=None):
        self.args = args
        if sampler is not None or args.evaluate:
            shuffle = False
        else:
            shuffle = True
        self.sampler = sampler
        self.rank = rank
        self.data = data
        self.device = device
        self.normalizer = normalizer
        self.optimizer = optimizer

        eff_bsz = args.batch_size / args.world_size
        self.update_after = math.ceil(eff_bsz / args.bsz_small)
        collator = Collator(args)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.bsz_small, shuffle=shuffle, num_workers=2, collate_fn=collator, pin_memory=True, sampler=sampler)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, epochs=args.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy='cos', pct_start=0.3)
        if checkpoint is not None and args.load_sch:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.align_loss = ContrastiveLoss(device)
        self.ce_loss = nn.CrossEntropyLoss()

    def evaluate(self, model):
        model.eval()
        for speechB, textB, textOut, _, logitLens, lmax, targetLens, keyL in tqdm(self.loader):
            if speechB is None:
                continue
            lmax = speechB.size(1)
            lens_norm = [1.*(x/lmax) for x in logitLens]
            speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=1000) # mean-var normalize
            speechB, logitLens = roll_in(speechB, logitLens, fac=self.args.roll_fac) # lower sequence length
            speechB, textB, textOut = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(textOut, self.device)
            with torch.no_grad():
                attn = model(speechB, textB, logitLens, targetLens, getW=True)
            attn_trim = attn[0]#[:,1:-1,:]
            attn_heads = attn_trim.cpu().numpy()
            fn = keyL[0]
            with open(f'{self.args.att_path}/{fn}.npy', 'wb') as f:
                np.save(f, attn_heads)

    def asr(self, model, logger):
        ##
        loss_asr_list = []
        loss_ctc_list = []
        ##
        for epoch in range(self.args.epochs_done+1, self.args.nepochs+1):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            print(f'Running epoch {epoch}.')
            step = 0
            ##
            loss_asr_rec = 0.
            loss_ctc_rec = 0.
            ##
            self.optimizer.zero_grad()
            for speechB, textB, textOut, textCtc, logitLens, lmax, targetLens, _ in tqdm(self.loader):
                if speechB is None:
                    continue
                model.train()
                step += 1
                lmax = speechB.size(1)
                lens_norm = [1.*(x/lmax) for x in logitLens]
                speechB = self.normalizer(speechB, torch.tensor(lens_norm).float(), epoch=epoch-1) # mean-var normalize
                speechB = inject_seqn(speechB) # sequence noise injection
                speechB = SpecDel(speechB, logitLens) # specaug --> del+ddel
                speechB, logitLens = roll_in(speechB, logitLens, fac=self.args.roll_fac) # lower sequence length
                speechB, textB, textOut, textCtc = load2gpu(speechB, self.device), load2gpu(textB, self.device), load2gpu(textOut, self.device), load2gpu(textCtc, self.device)
                if step % self.update_after != 0 and step != len(self.loader):
                    if self.args.ddp:
                        with model.no_sync():
                            logits = model(speechB, textB, logitLens, targetLens)
                            loss_asr = self.ce_loss(logits, textOut) / self.update_after
                            loss = loss_asr
                            if torch.isnan(loss):
                               logger.info(f'skipping batch no. {step} in epoch {epoch} due to NaN loss') 
                               continue
                            loss.backward()
                            loss_asr_rec += loss_asr.detach()
                    else:
                        logits, ctc_logits, logitLens = model(speechB, textB, logitLens, targetLens)
                        loss_ctc = self.ctc_loss(ctc_logits, textCtc, torch.tensor(logitLens), targetLens-1) / self.update_after 
                        loss_asr = torch.tensor(0.)#self.ce_loss(logits, textOut) / self.update_after
                        loss = (1. - self.args.ctc_wt) * loss_asr + self.args.ctc_wt * loss_ctc
                        if torch.isnan(loss):
                           logger.info(f'skipping batch no. {step} in epoch {epoch} due to NaN loss') 
                           continue
                        loss.backward()
                        ##
                        loss_asr_rec += loss_asr.detach()
                        loss_ctc_rec += loss_ctc.detach()
                        ##
                else:
                    logits, ctc_logits, logitLens = model(speechB, textB, logitLens, targetLens)
                    loss_ctc = self.ctc_loss(ctc_logits, textCtc, torch.tensor(logitLens), targetLens-1) / self.update_after 
                    loss_asr = torch.tensor(0.)#self.ce_loss(logits, textOut) / self.update_after
                    loss = (1. - self.args.ctc_wt) * loss_asr + self.args.ctc_wt * loss_ctc
                    if torch.isnan(loss):
                       logger.info(f'skipping batch no. {step} in epoch {epoch} due to NaN loss') 
                       continue
                    loss.backward()
                    ##
                    loss_asr_rec += loss_asr.detach()
                    loss_ctc_rec += loss_ctc.detach()
                    ##
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                    self.optimizer.step() 
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.args.ddp:
                        dist.all_reduce(loss_asr_rec)
                        loss_asr_list.append(loss_asr_rec.item() / dist.get_world_size())
                    else:
                        loss_asr_list.append(loss_asr_rec.item())
                        loss_ctc_list.append(loss_ctc_rec.item())
                    ##
                    loss_asr_rec = 0.
                    loss_ctc_rec = 0.
                    ##
                #if step % 256 == 0:
                #    print(f'| loss_asr = {np.mean(loss_asr_list)} | loss_ctc = {np.mean(loss_ctc_list)} | lr = {self.scheduler.get_last_lr()} |')
            if self.rank==0 or not self.args.ddp:
                log = f'| epoch = {epoch} | loss_asr = {np.mean(loss_asr_list)} | loss_ctc = {np.mean(loss_ctc_list)} | lr = {self.scheduler.get_last_lr()} |'
                print(log)
                logger.info(log)
            ##
            loss_asr_list = []
            loss_ctc_list = []
            ##
            if epoch % self.args.checkpoint_after == 0 and (self.rank==0 or not self.args.ddp):
                checkpoint = {'state_dict':model.state_dict(), 'normalizer':self.normalizer, 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'epochs_done':epoch}
                save_checkpoint(checkpoint, f'{self.args.save_path}')
