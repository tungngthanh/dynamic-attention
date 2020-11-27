#!/usr/bin/env bash

# Training
# TODO: 8 gpus run

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export ID=10
export fp16=1

export MAXTOKENS=8192
export UPDATE_FREQ=1

#export MAXTOKENS=16384
#export UPDATE_FREQ=1
export EXP=lightconv_wmt_en_de_big_v2

#export EXP=lightconv_wmt_en_de
#export ID=testing
bash train_ligh_conv.sh


# fixme: En-FR
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export ID=15
export fp16=1
#export MAXTOKENS=8192
#export UPDATE_FREQ=1
export MAXTOKENS=4096
export UPDATE_FREQ=2
export WORKERS=0

export EXP=lightconv_wmt_en_fr_big_v2
export PROBLEM=wmt14_en_fr

export CUDA_VISIBLE_DEVICES=6,7
export UPDATE_FREQ=2
bash train_ligh_conv.sh

export WORKERS=32

# fixme: En-Ru
#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1
#export ID=10
#export fp16=1
#export MAXTOKENS=8192
#export UPDATE_FREQ=2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ID=15
export fp16=1
export MAXTOKENS=8192
export UPDATE_FREQ=1
export MAXTOKENS=4096
export UPDATE_FREQ=2

export WORKERS=0
export EXP=lightconv_wmt_en_ru_big_v2
export PROBLEM=wmt14_en_ru
bash train_ligh_conv.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7
export ID=15
export fp16=0
export MAXTOKENS=4096
export UPDATE_FREQ=2
export WORKERS=0
export EXP=lightconv_wmt_en_ru_v2
export PROBLEM=wmt14_en_ru
bash train_ligh_conv.sh


# fixme: En-De
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=20v4
export fp16=1
#export MAXTOKENS=8192
#export UPDATE_FREQ=2
export MAXTOKENS=4096
export UPDATE_FREQ=1
export WORKERS=0
export EXP=lightconv_wmt_en_de_big_v2
export EXP=lightconv_wmt_en_de_big_v3
export EXP=lightconv_wmt_en_de
#export PROBLEM=wmt16_en_de_bpe32k
export PROBLEM=wmt16_en_de_new_bpe

bash train_ligh_conv.sh



# fixme: very big De-En WMT14 !
lightconv_wmt_en_de_big_128gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=20
export fp16=1
#export MAXTOKENS=8192
#export UPDATE_FREQ=2
export MAXTOKENS=3584
export UPDATE_FREQ=16
export WORKERS=0
export EXP=lightconv_wmt_en_de_big_128gpu
#export PROBLEM=wmt14_de_en
export PROBLEM=wmt14_en_de

bash train_ligh_conv.sh


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=20
export fp16=1
export MAXTOKENS=3584
export UPDATE_FREQ=16
export WORKERS=0
export EXP=lightconv_wmt_en_de_big_128gpu
#export PROBLEM=wmt14_de_en
export PROBLEM=wmt14_en_de_ext
export MAX_UPDATE=60000

#export ID=20u60k
bash train_ligh_conv.sh




# fixme: lgl
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ID=20testing
export fp16=1
#export MAXTOKENS=8192
#export UPDATE_FREQ=2
export MAXTOKENS=4096
export UPDATE_FREQ=1
export WORKERS=0
export EXP=lgl_hardlocal_wmt_en_de
export PROBLEM=wmt16_en_de_new_bpe

bash train_lgl_model.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
export ID=20testing
export fp16=1
export MAXTOKENS=4096
export UPDATE_FREQ=1
export WORKERS=0
export EXP=lgl_hardlocalavg_wmt_en_de
export PROBLEM=wmt16_en_de_new_bpe

bash train_lgl_model.sh


SAVE="save/dynamic_conv_wmt16en2de"
mkdir -p $SAVE
python -m torch.distributed.launch \
	--nproc_per_node 8 \
	train.py \
	data-bin/wmt16_en_de_bpe32k \
	--fp16  \
	--log-interval 100 \
	--no-progress-bar \
	--max-update 30000 \
	--share-all-embeddings \
	--optimizer adam \
	--adam-betas '(0.9, 0.98)' \
	--lr-scheduler inverse_sqrt \
	--clip-norm 0.0 \
	--weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--min-lr 1e-09 \
	--update-freq 16 \
	--attention-dropout 0.1 \
	--keep-last-epochs 10 \
	--ddp-backend=no_c10d \
	--max-tokens 3584 \
	--lr-scheduler cosine \
	--warmup-init-lr 1e-7 \
	--warmup-updates 10000 \
	--lr-shrink 1 \
	--max-lr 0.001 \
	--lr 1e-7 \
	--min-lr 1e-9 \
	--warmup-init-lr 1e-07 \
	--t-mult 1 \
	--lr-period-updates 20000 \
	--arch lightconv_wmt_en_de_big \
	--save-dir $SAVE \
	--dropout 0.3 \
	--attention-dropout 0.1 \
	--weight-dropout 0.1 \
	--encoder-glu 1 \
	--decoder-glu 1 \


# Evaluation
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/wmt16.en-de.joined-dict.newstest2014 --path "${SAVE}/checkpoint_best.pt" --batch-size 128 --beam 5 --remove-bpe --lenpen 0.5 --gen-subset test > wmt16_gen.txt
bash scripts/compound_split_bleu.sh wmt16_gen.txt











