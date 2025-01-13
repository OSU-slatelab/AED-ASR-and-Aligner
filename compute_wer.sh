path=sbCONT_LASpaper_timit_libri960_res40_char
logp=decodes/${path}_dev_other_noLP
rm -vrf ${logp}/full.txt
cat ${logp}/{0..3}.txt > "${logp}/full.txt"
python evaluate.py --path "${logp}/full.txt"
