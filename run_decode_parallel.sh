path=sbCONT_nhead4_LASpaper_libri960_res40_char
logp=decodes/${path}_dev_other_LP
world_size=6
rm -vrf ${logp}/*.log
rm -vrf ${logp}/*.txt
#python decode.py \
python -m torch.distributed.launch --nproc_per_node=$world_size decode.py \
	--test-path '/research/nfs_fosler_1/vishal/text/libri/dev_other.csv' \
	--world-size $world_size \
	--decode-path "${logp}" \
	--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/${path}.pth.tar" \
	--attn-type "content" \
	--length-norm \
	--nspeech-feat 80 \
	--sample-rate 16000 \
	--nhead 4 \
	--corpus 'librispeech' \
	--beam-size 16
rm -vrf ${logp}/full.txt
cat ${logp}/{0..5}.txt > "${logp}/full.txt"
python evaluate.py --path "${logp}/full.txt"
