python main.py \
	--nnodes 1 \
	--gpus 1 \
	--node_rank 0 \
	--gpu-num 1 \
	--roll-fac 1 \
	--valid-path '/research/nfs_fosler_1/vishal/audio/timit/test.csv'\
	--logging-file "logs/valid.log" \
	--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/sbCONT_LASpaper_timit_libri960_res40_char.pth.tar" \
	--batch-size 1 \
	--bsz-small 1 \
	--nspeech-feat 80 \
	--sample-rate 16000 \
	--nhead 1 \
	--attn-type "content" \
	--corpus 'timit' \
	--att-path 'timit_las' \
	--evaluate \
	--load-norm
#'/research/nfs_fosler_1/vishal/text/libri/dev_other.csv' \
#'/research/nfs_fosler_1/vishal/audio/timit/test.csv'\
