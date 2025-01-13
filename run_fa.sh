for i in 1
do
	for j in 0.01
	do
		python main.py \
			--nnodes 1 \
			--gpus 1 \
			--node_rank 0 \
			--gpu-num 1 \
			--roll-fac 1 \
			--valid-path '/research/nfs_fosler_1/vishal/text/readr/test1_manual.csv' \
			--gt-path '/research/nfs_fosler_1/vishal/text/readr/test1_align_manual.csv' \
			--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/ctc_corner_libri960_readr_res40_char.pth.tar" \
			--attn-type "content" \
			--logging-file "logs/readr_fa_ctc.log" \
			--batch-size 1 \
			--bsz-small 1 \
			--nspeech-feat 80 \
			--sample-rate 16000 \
			--nhead 1 \
			--res-fa 40 \
			--thres-fa $j \
			--corpus 'readr' \
			--mfa-align-path '/research/nfs_fosler_1/vishal/MFA/readr/alignments_test_flat1.json' \
			--force-align-ctc \
			--offset-fa \
			--load-norm
	done
done
#--valid-path '/research/nfs_fosler_1/vishal/text/libri/train_full_960.csv' \
#--gt-path '/research/nfs_fosler_1/vishal/audio/timit/test_align.csv' \
#--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/ctc_corner_libri960_res40_char.pth.tar" \
#--attn-type "content" \
#--logging-file "logs/libri_mfa_fa.log" \
#--batch-size 1 \
#--bsz-small 1 \
#--nspeech-feat 80 \
#--sample-rate 16000 \
#--nhead 1 \
#--res-fa 40 \
#--thres-fa $j \
#--corpus 'librispeech' \
#--mfa-align-path '/research/nfs_fosler_1/vishal/text/libri-alignments/alignments_train.json' \
#--offset-fa \
#--force-align-mfa \
#--cache-gt \
#--load-norm
