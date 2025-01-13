from speechbrain.utils.edit_distance import wer_details_by_utterance as wer_utt
from speechbrain.utils.edit_distance import wer_summary as wer_summ
import argparse
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='')

    args = parser.parse_args()

    hyp, ref = {}, {}
    #i = 0
    with open(args.path, 'r') as f:
        for line in f:
            i, r, h = line.strip().split('---->')
            r, h = r.strip().split(), h.strip().split()
            if len(r) == 0:
                continue
            hyp[f'utt{i}'] = h
            ref[f'utt{i}'] = r
    with open(args.path, 'a') as f:
        f.write(f'{wer_summ(wer_utt(ref, hyp))}')

if __name__ == '__main__':
    main()
