#!/usr/bin/env bash



# base-transformer from ntu
export CUDA_VISIBLE_DEVICES=6
export ID=30
export fp16=0
export UPDATE_FREQ=8
export MAXTOKENS=4096
#export INFERMODE=best
export MAX_UPDATE=110000
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=transformer_base
#export ARCH=restransformer_enc_logit_l2_wmt_en_de
#export ARCH=restransformer_enc_logit_l1_prog_wmt_en_de
#export ARCH=restransformer_enc_logitdense_l1_wmt_en_de
#export ARCH=restransformer_enc_logit_l2_prog_wmt_en_de
#export ARCH=restransformer_enc_logitsub_l1_prog_wmt_en_de  # seems already finished before
#export ARCH=restransformer_enc_logitnorm_l1_prog_wmt_en_de  # seems already finished before
export ARCH=restransformer_enc_logitsub_l2_prog_wmt_en_de  # seems already finished before
export ARCH=restransformer_enc_logitnorm_l2_prog_wmt_en_de  # seems already finished before
export ARCH=restransformer_enc_logitdensenorm_l2_wmt_en_de  # seems already finished before
export ARCH=restransformer_enc_logitdensesub_l2_wmt_en_de  # seems already finished before

bash train_docker_ntu.sh

export CUDA_VISIBLE_DEVICES=5
export ID=30
export fp16=0
export UPDATE_FREQ=8
export MAXTOKENS=4096
export MAX_UPDATE=100000
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=transformer_base
export ARCH=fi_transformer_wmt_en_de  # seems already finished before
bash train_docker_ntu.sh


export TASK=bertsrc_translation
export CRITERION=bertsrc_label_smoothed_cross_entropy
export ARCH=bertsrc_transformer_wmt_en_de

bash train_docker_ntu.sh


# todo:  DGX 128-GPUS Residual-cross attention

# ----- todo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=30
export fp16=1
export MAX_UPDATE=35000
export MAXTOKENS=3584
export UPDATE_FREQ=16
export INFERMODE=best
export LENPEN=0.5
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=lightconv_big
export ARCH=reslightconv_logitdense_l1_wmt_en_de_big
bash train_v2.sh


# bad
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=30
export fp16=1
export MAX_UPDATE=70000
export MAXTOKENS=4096
export UPDATE_FREQ=8
export INFERMODE=best
export LENPEN=0.5
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=lightconv_big
export ARCH=reslightconv_logitdense_l1_wmt_en_de_big
bash train_v2.sh



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=30
export fp16=1
export MAX_UPDATE=70000
export MAXTOKENS=4096
export UPDATE_FREQ=8
export INFERMODE=best
export LENPEN=0.5
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=lightconv_big
export ARCH=reslightconv_logit_prog_l1_wmt_en_de_big
bash train_v2.sh


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=30
export fp16=1
export MAX_UPDATE=35000
export MAXTOKENS=4096
export UPDATE_FREQ=16
export INFERMODE=best
export LENPEN=0.5
export PROBLEM=translate_ende_wmt16_bpe32k
export HPARAMS=lightconv_big
export ARCH=reslightconv_logit_prog_l1_wmt_en_de_big
bash train_v2.sh


