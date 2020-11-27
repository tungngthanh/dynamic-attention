#!/usr/bin/env bash

set -e

#pip install fairseq tensorflow-gpu tensor2tensor

# todo: specify gpus
[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}

export ROOT_DIR=`pwd`
export PROJDIR=tree_fairseq
export ROOT_DIR="${ROOT_DIR/\/tree_fairseq\/runs/}"

export user_dir=${ROOT_DIR}/${PROJDIR}


#export PROBLEM="${PROBLEM:-wmt16_en_de_new_bpe}"
export PROBLEM="${PROBLEM:-translate_ende_wmt_bpe32k}"

export RAW_DATA_DIR=${ROOT_DIR}/raw_data_fairseq/${PROBLEM}
export DATA_DIR=${ROOT_DIR}/data_fairseq/${PROBLEM}
export TRAIN_DIR_PREFIX=${ROOT_DIR}/train_fairseq_v2/${PROBLEM}


export ID="${ID:-1}"
export HPARAMS="${HPARAMS:-transformer_base}"
#export HPARAMS=transformer_big
#export HPARAMS=lightconv_big

[ -z "$ARCH" ] && { echo "Must set export ARCH="; exit 1; } || echo "ARCH = ${ARCH}"


if [ ${HPARAMS} == "transformer_base" ]; then
    export TASK="${TASK:-translation}"

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
#	export
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-4000}"
	export LR=0.0007
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.1}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-8}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"


elif [ ${HPARAMS} == "transformer_big" ]; then
    export TASK="${TASK:-translation}"

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-4000}"
	export LR=0.0005
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.3}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-8}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"

#	--arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#      --lr 0.0005 --min-lr 1e-09 \
#      --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#      --max-tokens 3584 \
#      --fp16

elif [ ${HPARAMS} == "transformer_big_128" ]; then
    export TASK="${TASK:-translation}"

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-4000}"
#	export LR="${LR:-0.0005}"
	export LR="${LR:-0.001}"
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.3}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
#	export MAXTOKENS="${MAXTOKENS:-3584}" # 8gpus
    export MAXTOKENS="${MAXTOKENS:-5120}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
	export MAX_UPDATE="${MAX_UPDATE:-42000}"


elif [ ${HPARAMS} == "transformer_big_128_rmsrceos" ]; then
    export TASK="${TASK:-seq2seq}"

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrtgpu
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-4000}"
#	export LR="${LR:-0.0005}"
	export LR="${LR:-0.001}"
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.3}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
#	export MAXTOKENS="${MAXTOKENS:-3584}" # 8gpus
    export MAXTOKENS="${MAXTOKENS:-5120}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
	export MAX_UPDATE="${MAX_UPDATE:-42000}"
	export rm_srceos="${rm_srceos:-1}"


#	--arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#      --lr 0.0005 --min-lr 1e-09 \
#      --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#      --max-tokens 3584 \
#      --fp16


elif [ ${HPARAMS} == "transformer_big_enfr" ]; then
    export TASK="${TASK:-translation}"

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=inverse_sqrt
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP="${WARMUP:-4000}"
	export LR=0.0007
	export MIN_LR=1e-09
	export DROPOUT="${DROPOUT:-0.1}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-8}"
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"


elif [ ${HPARAMS} == "lightconv_big" ]; then
    export TASK="${TASK:-translation}"

    export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=cosine
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=10000
	#export LR=0.0007
	export LR=1e-7
	export MIN_LR=1e-09
	export MAX_LR=0.001
	export DROPOUT="${DROPOUT:-0.3}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"
	export MAXTOKENS="${MAXTOKENS:-3584}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
	export DDP_BACKEND=no_c10d
	export MAX_UPDATE="${MAX_UPDATE:-30000}"
	export LRSRINK=1
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"

#
#elif [ ${HPARAMS} == "lightconv_base" ]; then
#    export TASK="${TASK:-translation}"
#
#    export OPTIM=adam
#	export ADAMBETAS='(0.9, 0.98)'
#	export CLIPNORM=0.0
#	export LRSCHEDULE=cosine
#	export WARMUP_INIT=1e-07
#	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
##	export WARMUP=10000
#	export WARMUP=4000
#	#export LR=0.0007
#	export LR=1e-7
#	export MIN_LR=1e-09
#	export MAX_LR=0.001
#	export DROPOUT="${DROPOUT:-0.1}"
#	export WDECAY=0.0
#	export LB_SMOOTH=0.1
#	export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"
#	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
#	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
#	export DDP_BACKEND=no_c10d
#	export MAX_UPDATE="${MAX_UPDATE:-30000}"
#	export LRSRINK=1
#	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"



elif [ ${HPARAMS} == "lightconv_big_enfr" ]; then
    export TASK="${TASK:-translation}"

    export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=cosine
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=10000
	#export LR=0.0007
	export LR=1e-7
	export MIN_LR=1e-09
	export MAX_LR=0.001
	export DROPOUT="${DROPOUT:-0.1}"
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"
	export MAXTOKENS="${MAXTOKENS:-3584}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
	export DDP_BACKEND=no_c10d
	export MAX_UPDATE="${MAX_UPDATE:-30000}"
	export LRSRINK=1
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"


elif [ ${HPARAMS} == "transformer_lightconv_big" ]; then
    export TASK="${TASK:-translation}"

    export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=cosine
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=10000
	#export LR=0.0007
	export LR=1e-7
	export MIN_LR=1e-09
	export MAX_LR=0.001
	export DROPOUT=0.3
	export WDECAY=0.0
	export LB_SMOOTH=0.1
#	export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-70000}"
	export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"
	export MAXTOKENS="${MAXTOKENS:-3584}" # 8gpus
	export UPDATE_FREQ="${UPDATE_FREQ:-16}"
	export DDP_BACKEND=no_c10d
	export MAX_UPDATE="${MAX_UPDATE:-30000}"
	export LRSRINK=1
	export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"

elif [ ${HPARAMS} == "lightconv_lm_gbw" ]; then

#	export MODEL=lightconv
#	export HPARAMS=lm_gbw
	export ARCH=lightconv_lm_gbw
	export TASK=translation

	export OPTIM=adam
	export ADAMBETAS='(0.9, 0.98)'
	export CLIPNORM=0.0
	export LRSCHEDULE=cosine
	export WARMUP_INIT=1e-07
	# wamrup 4000 for 8 gpus, 16000 for 1 gpus
	export WARMUP=16000
	#export LR=0.0007
	export LR=1e-7
	export MIN_LR=1e-09
	export MAX_LR=0.001
	export DROPOUT=0.1
	export WDECAY=0.0
	export LB_SMOOTH=0.1
	export MAXTOKENS="${MAXTOKENS:-4096}" # 8gpus
	export UPDATE_FREQ=4
	export DDP_BACKEND=no_c10d
	export MAX_UPDATE=100000
	export LRSRINK=1

	echo "lightconv_lm_gbw not work yet"


elif [ ${HPARAMS} == "transformer_base_lm_gbw" ]; then

#	export MODEL=transformer
#	export HPARAMS=lm_gbw
#	export ARCH=${MODEL}_${HPARAMS}
	export ARCH=transformer_lm
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
	export UPDATE_FREQ="${UPDATE_FREQ:-4}"
	export MAX_UPDATE=100000

else
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
	export UPDATE_FREQ="${UPDATE_FREQ:-2}"
	export MAX_UPDATE=100000
#    echo "undefined HPARAMS: ${HPARAMS}"
#    exit 1
fi


export LR_PERIOD_UPDATES="${LR_PERIOD_UPDATES:-20000}"

export MAX_UPDATE="${MAX_UPDATE:-150000}"
export KEEP_LAS_CHECKPOINT="${KEEP_LAS_CHECKPOINT:-10}"

export DDP_BACKEND="${DDP_BACKEND:-c10d}"
export LRSRINK="${LRSRINK:-0.1}"
export MAX_LR="${MAX_LR:-0.001}"
export WORKERS="${WORKERS:-0}"
export INFER="${INFER:-y}"
export DISTRIBUTED="${DISTRIBUTED:-y}"

export CRITERION="${CRITERION:-label_smoothed_cross_entropy}"

export fp16="${fp16:-0}"
export rm_srceos="${rm_srceos:-0}"
export rm_lastpunct="${rm_lastpunct:-0}"
export nobar="${nobar:-1}"

# todo: specify distributed and fp16
#[ ${fp16} -eq 0 ] && export fp16s="#" || export fp16s=
[ ${fp16} -eq 1 ] && export fp16s="--fp16 " || export fp16s=
[ ${rm_srceos} -eq 1 ] && export rm_srceos_s="--remove-eos-from-source " || export rm_srceos_s=
[ ${rm_lastpunct} -eq 1 ] && export rm_lastpunct_s="--remove-last-punct-source " || export rm_lastpunct_s=
[ ${nobar} -eq 1 ] && export nobarstr="--no-progress-bar" || export nobarstr=
[ ${NUM_GPU} -gt 1 ] && export distro= || export distro="#"


export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
export log_interval="${log_interval:-100}"

export TRAIN_DIR=${TRAIN_DIR_PREFIX}/${ARCH}-${HPARAMS}-b${MAXTOKENS}-gpu${NUM_GPU}-upfre${UPDATE_FREQ}-${fp16}fp16-id${ID}

#rm -rf ${TRAIN_DIR}
echo "====================================================="
echo "START TRAINING: ${TRAIN_DIR}"
echo "PROJDIR: ${PROJDIR}"
echo "user_dir: ${user_dir}"
echo "ARCH: ${ARCH}"
echo "HPARAMS: ${HPARAMS}"
echo "DISTRO: ${distro}"
echo "INFER: ${INFER}"
echo "CRITERION: ${CRITERION}"
echo "fp16: ${fp16}"
echo "rm_srceos: ${rm_srceos}"
echo "rm_lastpunct_s: ${rm_lastpunct_s}"
echo "====================================================="

mkdir -p ${TRAIN_DIR}

if [ ${DISTRIBUTED} == "y" ]; then
    if [ ${NUM_GPU} -gt 1 ]; then
        export init_command="python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} $(which fairseq-train) ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} "
    else
        export init_command="$(which fairseq-train) ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} "
    fi
else
    export init_command="$(which fairseq-train) ${DATA_DIR} --ddp-backend=${DDP_BACKEND} ${nobarstr} "
fi

echo "init_command = ${init_command}"

#	${fp16s} --fp16

echo "Run model ${ARCH}, ${HPARAMS}"

if [ ${HPARAMS} == "lightconv_big" -o ${HPARAMS} == "lightconv_big_enfr" ]; then
#    eval "${init_command} \
#	--user-dir ${user_dir} \
#	--arch ${ARCH} \
#	--task ${TASK} \
#	--log-interval ${log_interval} \
#	--num-workers ${WORKERS} \
#	--share-all-embeddings \
#	--optimizer ${OPTIM} \
#	--clip-norm ${CLIPNORM} \
#	--lr-scheduler ${LRSCHEDULE} \
#	--warmup-init-lr  ${WARMUP_INIT} \
#	--warmup-updates ${WARMUP} \
#	--lr-shrink ${LRSRINK} \
#	--lr ${LR} \
#	--min-lr ${MIN_LR} \
#	--max-lr ${MAX_LR} \
#	--t-mult 1 \
#	--lr-period-updates ${LR_PERIOD_UPDATES} \
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
#	--attention-dropout 0.1 \
#	--weight-dropout 0.1 \
#	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
#	--encoder-glu 1 \
#	--decoder-glu 1 \
#	${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s}"
    echo "Run operation: HPARAMS == lightconv_big -o HPARAMS == lightconv_big_enfr"

    eval "${init_command} \
        --user-dir ${user_dir} \
        --save-dir ${TRAIN_DIR} \
        --arch ${ARCH} \
        --task ${TASK} \
        --log-interval ${log_interval} \
        --max-update ${MAX_UPDATE} \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --lr-scheduler inverse_sqrt \
        --clip-norm ${CLIPNORM} \
        --weight-decay ${WDECAY} \
        --criterion ${CRITERION} \
        --label-smoothing ${LB_SMOOTH} \
        --min-lr 1e-09 \
        --update-freq ${UPDATE_FREQ} \
        --attention-dropout 0.1 \
        --keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
        --ddp-backend=no_c10d \
        --max-tokens ${MAXTOKENS} \
        --lr-scheduler cosine \
        --warmup-init-lr  ${WARMUP_INIT} \
        --warmup-updates ${WARMUP} \
        --lr-shrink ${LRSRINK} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --max-lr ${MAX_LR} \
        --warmup-init-lr  ${WARMUP_INIT} \
        --t-mult 1 \
        --lr-period-updates ${LR_PERIOD_UPDATES} \
        --dropout ${DROPOUT} \
        --attention-dropout 0.1 \
        --weight-dropout 0.1 \
        --encoder-glu 1 \
        --decoder-glu 1 \
        --left-pad-source ${LEFT_PAD_SRC} \
        ${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s}"


#    elif [ ${HPARAMS} == "transformer_lightconv_big" ]; then
#        eval "${init_command} \
#        --user-dir ${user_dir} \
#        --arch ${ARCH} \
#        --task ${TASK} \
#        --log-interval ${log_interval} \
#        --num-workers ${WORKERS} \
#        --share-all-embeddings \
#        --optimizer ${OPTIM} \
#        --clip-norm ${CLIPNORM} \
#        --lr-scheduler ${LRSCHEDULE} \
#        --warmup-init-lr  ${WARMUP_INIT} \
#        --warmup-updates ${WARMUP} \
#        --lr-shrink ${LRSRINK} \
#        --lr ${LR} \
#        --min-lr ${MIN_LR} \
#        --max-lr ${MAX_LR} \
#        --t-mult 1 \
#        --lr-period-updates ${LR_PERIOD_UPDATES} \
#        --dropout ${DROPOUT} \
#        --weight-decay ${WDECAY} \
#        --update-freq ${UPDATE_FREQ} \
#        --criterion ${CRITERION} \
#        --label-smoothing ${LB_SMOOTH} \
#        --adam-betas '(0.9, 0.98)' \
#        --max-tokens ${MAXTOKENS} \
#        --left-pad-source ${LEFT_PAD_SRC} \
#        --max-update ${MAX_UPDATE} \
#        --save-dir ${TRAIN_DIR} \
#        --keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
#        ${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s}"

#--attention-dropout 0.1 \
#	--weight-dropout 0.1 \
#	--encoder-glu 1 \
#	--decoder-glu 1 \

elif [ ${HPARAMS} == "transformer_base_lm_gbw" ]; then
    echo "Run operation: HPARAMS == transformer_base_lm_gbw -o HPARAMS == lm_gbw"
    eval "${init_command} \
	--arch ${ARCH} \
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
	--dropout ${DROPOUT} \
	--weight-decay ${WDECAY} \
	--update-freq ${UPDATE_FREQ} \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.9, 0.98)' \
	--max-tokens ${MAXTOKENS} \
	--max-update ${MAX_UPDATE} \
	--save-dir ${TRAIN_DIR} \
	--keep-last-epochs ${KEEP_LAS_CHECKPOINT} \
	${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s}"

else
	eval "${init_command} \
	--user-dir ${user_dir} \
	--arch ${ARCH} \
	--task language_modeling \
	--log-interval ${log_interval} \
	--num-workers ${WORKERS} \
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
	--criterion ${CRITERION} \
	--label-smoothing ${LB_SMOOTH} \
	--adam-betas '(0.9, 0.98)' \
	--max-tokens ${MAXTOKENS} \
	--max-update 100000 \
	--save-dir ${TRAIN_DIR} \
	--save-interval-updates 10000 \
	--keep-interval-updates 20 \
	${fp16s}  ${rm_srceos_s} ${rm_lastpunct_s}"
fi


echo "=================="
echo "=================="
echo "finish training at ${TRAIN_DIR}"


if [ ${INFER} == "y" ]; then
    echo "Start inference ...."
    bash infer_model.sh
fi


