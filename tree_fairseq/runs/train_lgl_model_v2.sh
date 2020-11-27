#!/usr/bin/env bash


set -e

# todo: specify gpus
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}


# specify machines
export ROOT_DIR_NTU=/data/nxphi47/projects/tree
export ROOT_DIR_NTU_DOCKER=/nxphi47/projects/tree
export ROOT_DIR_DGX=/home/i2r/xlli/scratch/fifi/projects/nmt
export ROOT_DIR_DGX_DOCKER=/projects/nmt
export ROOT_DIR_HINTON=/media/sbmaruf/backup/all_phi/projects/nmt
export ROOT_DIR_HOME="/media/ttnguyen/New Volume/Projects/nmt"
#export ROOT_DIR=/home/i2r/xlli/scratch/fifi/projects/nmt
#export ROOT_DIR=/data/nxphi47/projects/tree

#export MACHINE="${MACHINE:-ntu}"
export ROOT_DIR=${ROOT_DIR_HOME}

echo "MACHINE -> ${MACHINE}"

#if [ ${MACHINE} == "ntu" ]; then
#	export ROOT_DIR=${ROOT_DIR_NTU}
#elif [ ${MACHINE} == "ntu_docker" ]; then
#	export ROOT_DIR=${ROOT_DIR_NTU_DOCKER}
#elif [ ${MACHINE} == "hinton" ]; then
#	export ROOT_DIR=${ROOT_DIR_HINTON}
#
#elif [ ${MACHINE} == "dgx" ]; then
#	export ROOT_DIR=${ROOT_DIR_DGX}
#elif [ ${MACHINE} == "dgx_docker" ]; then
#	export ROOT_DIR=${ROOT_DIR_DGX_DOCKER}
#
#else
#	echo "MACHINE not found: ${MACHINE}"
#	export DATA_DIR_ROOT="${DATA_DIR_ROOT:-$DATA_DIR_ROOT_NTU}"
#	export TRAIN_DIR_ROOT="${TRAIN_DIR_ROOT:-$TRAIN_DIR_ROOT_NTU}"
#fi

export ROOT_DIR=`pwd`
export PROJDIR=tree_fairseq
#export ROOT_DIR="${ROOT_DIR/\/tree_fairseq/}"
export ROOT_DIR="${ROOT_DIR/\/tree_fairseq\/runs/}"

export user_dir=${ROOT_DIR}/${PROJDIR}

#export EPOCHS="${EPOCHS:-300}"
export PROBLEM="${PROBLEM:-wmt16_en_de_new_bpe}"

export RAW_DATA_DIR=${ROOT_DIR}/raw_data_fairseq/${PROBLEM}
export DATA_DIR=${ROOT_DIR}/data_fairseq/${PROBLEM}
export TRAIN_DIR_PREFIX=${ROOT_DIR}/train_fairseq_v2/${PROBLEM}

export EXP="${EXP:-lgl_hardlocal_wmt_en_de}"
export ID="${ID:-1}"

if [ ${EXP} == "transformer_wmt_ende_8gpu1" ]; then

	export MODEL=transformer
	export HPARAMS=wmt_en_de
	export ARCH=${MODEL}_${HPARAMS}
	export TASK=translation

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=4000
	export LR=0.0007
	export MIN_LR=1e-09
	export DROPOUT=0.1
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ=8

elif [ ${EXP} == "lgl_hardlocal_wmt_en_de" ]; then

    export MODEL=lgl
    export HPARAMS=hardlocal_wmt_en_de
    export ARCH=lgl_hardlocal_wmt_en_de
    export TASK=translation

    export OPTIM=adam
    export ADAMBETAS='(0.9, 0.98)'
    export CLIPNORM=0.0
    export LRSCHEDULE=inverse_sqrt
    export WARMUP_INIT=1e-07
    # wamrup 4000 for 8 gpus, 16000 for 1 gpus
    export WARMUP=4000
    export LR=0.0007
    export MIN_LR=1e-09
    export DROPOUT=0.1
    export WDECAY=0.0
    export LB_SMOOTH=0.1
    export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-8}"
	export LEFT_PAD_SRC=True


elif [ ${EXP} == "lgl_hardlocalavg_wmt_en_de" ]; then

    export MODEL=lgl
    export HPARAMS=hardlocalavg_wmt_en_de
    export ARCH=lgl_hardlocalavg_wmt_en_de
    export TASK=translation

    export OPTIM=adam
    export ADAMBETAS='(0.9, 0.98)'
    export CLIPNORM=0.0
    export LRSCHEDULE=inverse_sqrt
    export WARMUP_INIT=1e-07
    # wamrup 4000 for 8 gpus, 16000 for 1 gpus
    export WARMUP=4000
    export LR=0.0007
    export MIN_LR=1e-09
    export DROPOUT=0.1
    export WDECAY=0.0
    export LB_SMOOTH=0.1
    export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-8}"
	export LEFT_PAD_SRC=True


# FIXME -- big models-----

elif [ ${EXP} == "transformer_vaswani_wmt_en_de_big_8gpu1" ]; then

	export MODEL=transformer
	export HPARAMS=vaswani_wmt_en_de_big
	export ARCH=${MODEL}_${HPARAMS}
	export TASK=translation
	
	export OPTIM=adam
	export ADAMBETAS="(0.9, 0.98)"
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=4000
	export LR=0.0005
	export MIN_LR=1e-09
	export DROPOUT=0.3
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	
	
	export MAXTOKENS=3584 # 8gpus
	export UPDATE_FREQ=8


else
	echo "EXP not found: ${EXP}"
	exit 0
fi


# default parmas
export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"

export MAX_UPDATE="${MAX_UPDATE:-1000000}"
export KEEP_LAS_CHECKPOINT="${KEEP_LAS_CHECKPOINT:-20}"
export DDP_BACKEND="${DDP_BACKEND:-c10d}"
export LRSRINK="${LRSRINK:-0.1}"
export MAX_LR="${MAX_LR:-0.001}"
export WORKERS="${WORKERS:-16}"

export fp16="${fp16:-0}"
export nobar="${nobar:-1}"

#if [ ${fp16} -eq 0 ]; then
#	export fp16s="#"
#else
#	export fp16s=
#fi

#[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
#if [ ${NUM_GPU} -gt 1 ]; then
#	export distro="#"

# todo: specify distributed and fp16
[ ${fp16} -eq 0 ] && export fp16s="#" || export fp16s=
[ ${nobar} -eq 1 ] && export nobarstr="--no-progress-bar" || export nobarstr=
[ ${NUM_GPU} -gt 1 ] && export distro= || export distro="#"


export LEFT_PAD_SRC="${LEFT_PAD_SRC:-True}"
export log_interval="${log_interval:-150}"


# ACTUAL TRAINING
#export TRAIN_DIR=${TRAIN_DIR_PREFIX}/${MODEL}-${HPARAMS}-b${MAXTOKENS}-gpu${NUM_GPU}-upfre${UPDATE_FREQ}
export TRAIN_DIR=${TRAIN_DIR_PREFIX}/${ARCH}-b${MAXTOKENS}-gpu${NUM_GPU}-upfre${UPDATE_FREQ}-${fp16}fp16-id${ID}

#rm -rf ${TRAIN_DIR}
echo "====================================================="
echo "START TRAINING: ${TRAIN_DIR}"
echo "PROJDIR: ${PROJDIR}"
echo "user_dir: ${user_dir}"
echo "EXP: ${EXP}"
echo "DISTRO: ${distro}"
echo "fp16: ${fp16}"
echo "fp16s: ${fp16s}"
echo "====================================================="

mkdir -p ${TRAIN_DIR}


# start running
#if [ ${NUM_GPU} -gt 1 ]; then
#	export init_command="python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} train.py ${DATA_DIR} --ddp-backend=no_c10d ${nobarstr} "
#else
#	export init_command="python train.py ${DATA_DIR} ${nobarstr} "
#fi
if [ ${NUM_GPU} -gt 1 ]; then
	export init_command="python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} $(which fairseq-train) ${DATA_DIR} --ddp-backend=no_c10d ${nobarstr} "
else
	export init_command="$(which fairseq-train) ${DATA_DIR} ${nobarstr} "
fi

#export init_command="python train.py ${DATA_DIR} ${nobarstr} "

echo "commend: ${init_command}"

#	--source_lang en \

if [ ${HPARAMS} == "lm_gbw" ]; then
	echo "Run model ${MODEL}, ${ARCH}, ${HPARAMS}"
	eval "${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task ${TASK} \
	--task language_modeling \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
	--optimizer ${OPTIM} \
	--clip-norm ${CLIPNORM} \
	--lr-scheduler ${LRSCHEDULE} \
	--warmup-init-lr  ${WARMUP_INIT} \
	--warmup-updates ${WARMUP} \
	--lr-shrink ${LRSRINK} \
	--lr ${LR} \
	--min-lr ${MIN_LR} \
	--max-lr ${MAX_LR} \
	--t-mult 1 \
	--lr-period-updates 20000 \
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.9, 0.98)' \
	--max-tokens ${MAXTOKENS} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--attention-dropout 0.1 \
	--weight-dropout 0.1 \
	--decoder-glu 1 \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	${fp16s} --fp16"

#	--left-pad-source ${LEFT_PAD_SRC} \

elif [ ${MODEL} == "lightconv" ]; then
	echo "Run model ${MODEL}, ${ARCH}, ${HPARAMS}"
	eval "${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task ${TASK} \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
	--share-all-embeddings \
	--optimizer ${OPTIM} \
	--clip-norm ${CLIPNORM} \
	--lr-scheduler ${LRSCHEDULE} \
	--warmup-init-lr  ${WARMUP_INIT} \
	--warmup-updates ${WARMUP} \
	--lr-shrink ${LRSRINK} \
	--lr ${LR} \
	--min-lr ${MIN_LR} \
	--max-lr ${MAX_LR} \
	--t-mult 1 \
	--lr-period-updates ${LR_PERIOD_UPDATES} \
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.9, 0.98)' \
	--max-tokens ${MAXTOKENS} \
	--left-pad-source ${LEFT_PAD_SRC} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--attention-dropout 0.1 \
	--weight-dropout 0.1 \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	--encoder-glu 1 \
	--decoder-glu 1 \
	${fp16s} --fp16"

#	--lr-period-updates 20000 \
#$LR_PERIOD_UPDATES


#	--encoder-glu 1 \
else
#python -m torch.distributed.launch --nproc_per_node 8

#	eval "python \
#	${distro} -m torch.distributed.launch --nproc_per_node ${NUM_GPU} \
#	train.py ${DATA_DIR} \
#	--no-progress-bar \
#	--arch ${ARCH} \
#	--task ${TASK} \
#	--num-workers ${WORKERS} \
#	--share-all-embeddings \
#	--optimizer ${OPTIM} \
#	--clip-norm ${CLIPNORM} \
#	--lr-scheduler ${LRSCHEDULE} \
#	--warmup-init-lr  ${WARMUP_INIT} \
#	--warmup-updates ${WARMUP} \
#	--lr ${LR} \
#	--min-lr ${MIN_LR} \
#	--dropout ${DROPOUT} \
#	--weight-decay ${WDECAY} \
#	--update-freq ${UPDATE_FREQ} \
#	--criterion label_smoothed_cross_entropy \
#	--label-smoothing ${LB_SMOOTH} \
#	--adam-betas '(0.9, 0.98)' \
#	--max-tokens ${MAXTOKENS} \
#	--left-pad-source ${LEFT_PAD_SRC} \
#	--max-update ${MAX_UPDATE} \
#	--save-dir ${TRAIN_DIR} \
#	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
#	${fp16s} --fp16"

#	${distro} -m torch.distributed.launch --nproc_per_node ${NUM_GPU} \

#	export init_command="--ddp-backend=no_c10d"
#	[ ${NUM_GPU} -gt 1 ] && export distro= || export distro="#"
	echo "Run model ${MODEL}, ${ARCH}, ${HPARAMS}"
	eval "${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task ${TASK} \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
	--share-all-embeddings \
	--optimizer ${OPTIM} \
	--clip-norm ${CLIPNORM} \
	--lr-scheduler ${LRSCHEDULE} \
	--warmup-init-lr  ${WARMUP_INIT} \
	--warmup-updates ${WARMUP} \
	--lr ${LR} \
	--min-lr ${MIN_LR} \
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.9, 0.98)' \
	--max-tokens ${MAXTOKENS} \
	--left-pad-source ${LEFT_PAD_SRC} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	${fp16s} --fp16"


fi


echo "=================="
echo "=================="
echo "finish training at ${TRAIN_DIR}"
