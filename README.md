# Attention-based ASR and alignment

This repository contains an implementation of attention based ASR using LSTMs. Further, this ASR model is used to build a Forced Aligner.

`run_fa.sh` will run a forced alignment using an AED ASR model saved at `--ckpt-path`.
* `--force-align` will run the AED based aligner
* `--force-align-ctc` will run the CTC based aligner
