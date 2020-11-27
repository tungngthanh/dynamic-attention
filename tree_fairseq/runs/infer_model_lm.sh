#!/usr/bin/env bash



set -e
# specify machines

[ -z "$CUDA_VISIBLE_DEVICES" ] && { echo "Must set export CUDA_VISIBLE_DEVICES="; exit 1; } || echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPU=${#GPUS[@]}

#export ROOT_DIR=/home/i2r/xlli/scratch/fifi/projects/nmt
#export ROOT_DIR=/data/nxphi47/projects/tree

export MACHINE="${MACHINE:-ntu}"

echo "MACHINE -> ${MACHINE}"

#if [ ${MACHINE} == "ntu" ]; then
#	export ROOT_DIR=${ROOT_DIR_NTU}
##elif [ ${MACHINE} == "hinton" ]; then
##elif [ ${MACHINE} == "hinton_sbmaruf" ]; then
##elif [ ${MACHINE} == "titanv" ]; then
#elif [ ${MACHINE} == "dgx" ]; then
#	export ROOT_DIR=${ROOT_DIR_DGX}
##elif [ ${MACHINE} == "dgx_dock" ]; then
#else
#	echo "MACHINE not found: ${MACHINE}"
#	export DATA_DIR_ROOT="${DATA_DIR_ROOT:-$DATA_DIR_ROOT_NTU}"
#	export TRAIN_DIR_ROOT="${TRAIN_DIR_ROOT:-$TRAIN_DIR_ROOT_NTU}"
#fi

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

#export ROOT_DIR=`pwd`
#export ROOT_DIR="${ROOT_DIR/\/fairseq_fifi/}"

export ROOT_DIR=`pwd`
export PROJDIR=tree_fairseq
export ROOT_DIR="${ROOT_DIR/\/tree_fairseq\/runs/}"


export user_dir=${ROOT_DIR}/${PROJDIR}



if [ -d ${TRAIN_DIR} ]; then
	# if train exists
    echo "directory train exists!: ${TRAIN_DIR}"
else
    echo "directory train not exists!: ${TRAIN_DIR}"
    exit 1
fi


#export EPOCHS="${EPOCHS:-300}"
#export PROBLEM="${PROBLEM:-wmt16_en_de_new_bpe}"
export PROBLEM="${PROBLEM:-translate_ende_wmt_bpe32k}"

export RAW_DATA_DIR=${ROOT_DIR}/raw_data_fairseq/${PROBLEM}
export DATA_DIR=${ROOT_DIR}/data_fairseq/${PROBLEM}
export TRAIN_DIR_PREFIX=${ROOT_DIR}/train_fairseq_v2/${PROBLEM}

export EXP="${EXP:-transformer_wmt_ende_8gpu1}"
export ID="${ID:-1}"
export INFER_ID="${INFER_ID:-1}"



export TGT_LANG="${TGT_LANG:-de}"
export SRC_LANG="${SRC_LANG:-en}"
export TESTSET="${TESTSET:-newstest2014}"


export INFERMODE="${INFERMODE:-avg}"
#export REF=${RAW_DATA_DIR}/${TESTSET}.tok.bpe.32000.${TGT_LANG}
#export INPUT=${RAW_DATA_DIR}/${TESTSET}.tok.bpe.32000.${SRC_LANG}
export INFER_DIR=${TRAIN_DIR}/infer
mkdir -p ${INFER_DIR}

# generate parameters

export TASK="${TASK:-translation}"
export BEAM="${BEAM:-5}"
#export INFER_BSZ="${INFER_BSZ:-128}"
#export INFER_BSZ="${INFER_BSZ:-4096}"
export INFER_BSZ="${INFER_BSZ:-2048}"
export LENPEN="${LENPEN:-0.6}"
#export LEFT_PAD_SRC="${LEFT_PAD_SRC:-True}"
export LEFT_PAD_SRC="${LEFT_PAD_SRC:-False}"
export RMBPE="${RMBPE:-y}"
export GETBLEU="${GETBLEU:-y}"
export NEWCODE="${NEWCODE:-y}"
export INFER_TASK="${INFER_TASK:-mt}"
export HEAT="${HEAT:-n}"
export rm_srceos="${rm_srceos:-0}"
export rm_lastpunct="${rm_lastpunct:-0}"
export get_entropies="${get_entropies:-0}"
export GENSET="${GENSET:-valid}"

export GEN_DIR=${INFER_DIR}/${GENSET}.tok.rmBpe${RMBPE}.genout.${TGT_LANG}.b${BEAM}.lenpen${LENPEN}.leftpad${LEFT_PAD_SRC}.${INFERMODE}


[ ${rm_srceos} -eq 1 ] && export rm_srceos_s="--remove-eos-from-source " || export rm_srceos_s=
[ ${rm_lastpunct} -eq 1 ] && export rm_lastpunct_s="--remove-last-punct-source " || export rm_lastpunct_s=
[ ${get_entropies} -eq 1 ] && export get_entropies_s="--layer-att-entropy " || export get_entropies_s=
[ ${RMBPE} == "y" ] && export rm_bpe_s="--remove-bpe " || export rm_bpe_s=

#/projects/nmt/train_fi_fairseq/wmt16_en_de_new_bpe/me_vaswani_wmt_en_de_big-transformer_big_128-b5120-gpu8-upfre16-1fp16-id24
echo "========== INFERENCE ================="
echo "TASK = ${TASK}"
echo "infermode = ${INFERMODE}"
echo "BEAM = ${BEAM}"
echo "INFER_BSZ = ${INFER_BSZ}"
echo "LENPEN = ${LENPEN}"
echo "LEFT_PAD_SRC = ${LEFT_PAD_SRC}"
echo "RMBPE = ${RMBPE}"
echo "GETBLEU = ${GETBLEU}"
echo "NEWCODE = ${NEWCODE}"
echo "rm_srceos = ${rm_srceos} - string=${rm_srceos_s}"
echo "rm_lastpunct = ${rm_lastpunct} - string=${rm_lastpunct_s}"
echo "========== INFERENCE ================="

# selecting infermode
# ---------------------------------------------------------------------------------------------------
if [ ${INFERMODE} == "best" ]; then

    export CHECKPOINT=${TRAIN_DIR}/checkpoint_best.pt
    mkdir -p ${GEN_DIR}

    export GEN_OUT=${GEN_DIR}/infer
    export HYPO=${GEN_OUT}.hypo
    export REF=${GEN_OUT}.ref
    export BLEU_OUT=${GEN_OUT}.bleu

    echo "GEN_OUT = ${GEN_OUT}"

#    python generate.py ${DATA_DIR} \
#        --path ${CHECKPOINT} \
#        --left-pad-source ${LEFT_PAD_SRC} \
#        --batch-size ${INFER_BSZ} \
#        --beam ${BEAM} \
#        --lenpen ${LENPEN} \
#        --remove-bpe \
#        | tee ${GEN_OUT}

# ---------------------------------------------------------------------------------------------------------
elif [ ${INFERMODE} == "avg" ]; then

    export AVG_NUM="${AVG_NUM:-5}"
#    export UPPERBOUND="${UPPERBOUND:-22}"
    export UPPERBOUND="${UPPERBOUND:-100000000}"
#    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$TRAIN_DIR/averaged_model.${AVG_NUM}.u${UPPERBOUND}.pt}"
    export LAST_EPOCH=`python get_last_checkpoint.py --dir=${TRAIN_DIR}`

    export GEN_DIR=${GEN_DIR}.avg${AVG_NUM}.e${LAST_EPOCH}.u${UPPERBOUND}
    mkdir -p ${GEN_DIR}

    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$GEN_DIR/averaged_model.id${INFER_ID}.avg${AVG_NUM}.e${LAST_EPOCH}.u${UPPERBOUND}.pt}"
    export GEN_OUT=${GEN_DIR}/infer
    export GEN_OUT=${GEN_OUT}.avg${AVG_NUM}
    export HYPO=${GEN_OUT}.hypo
    export REF=${GEN_OUT}.ref
    export BLEU_OUT=${GEN_OUT}.bleu

    echo "GEN_DIR = ${GEN_DIR}"
    echo "GEN_OUT = ${GEN_OUT}"


    echo "AVG_NUM = ${AVG_NUM}"
    echo "LAST_EPOCH = ${LAST_EPOCH}"
    echo "AVG_CHECKPOINT_OUT = ${AVG_CHECKPOINT_OUT}"
    echo "---- Score by averaging last checkpoints ${AVG_NUM} -> ${AVG_CHECKPOINT_OUT}"
    echo "Generating average checkpoints..."
#    exit 1

    if [ -f ${AVG_CHECKPOINT_OUT} ]; then
        echo "File ${AVG_CHECKPOINT_OUT} exists...."
    else
        python ../scripts/average_checkpoints.py \
        --user-dir ${user_dir} \
        --inputs ${TRAIN_DIR} \
        --num-epoch-checkpoints ${AVG_NUM} \
        --checkpoint-upper-bound ${UPPERBOUND} \
        --output ${AVG_CHECKPOINT_OUT}
        echo "Finish generating averaged, start generating samples"
    fi

    export CHECKPOINT=${AVG_CHECKPOINT_OUT}

elif [ ${INFERMODE} == "avg_ema" ]; then
    export AVG_NUM="${AVG_NUM:-5}"
    export EMA_DECAY="${EMA_DECAY:-0.9}"
#    export UPPERBOUND="${UPPERBOUND:-22}"
    export UPPERBOUND="${UPPERBOUND:-100000000}"
#    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$TRAIN_DIR/averaged_model.${AVG_NUM}.u${UPPERBOUND}.pt}"
    export LAST_EPOCH=`python get_last_checkpoint.py --dir=${TRAIN_DIR}`

    export GEN_DIR=${GEN_DIR}.avg${AVG_NUM}.ema${EMA_DECAY}.e${LAST_EPOCH}.u${UPPERBOUND}
    mkdir -p ${GEN_DIR}

    export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$GEN_DIR/averaged_model.id${INFER_ID}.avg${AVG_NUM}.ema${EMA_DECAY}.e${LAST_EPOCH}.u${UPPERBOUND}.pt}"
    export GEN_OUT=${GEN_DIR}/infer
    export GEN_OUT=${GEN_OUT}.avg${AVG_NUM}
    export HYPO=${GEN_OUT}.hypo
    export REF=${GEN_OUT}.ref
    export BLEU_OUT=${GEN_OUT}.bleu

    echo "GEN_DIR = ${GEN_DIR}"
    echo "GEN_OUT = ${GEN_OUT}"


    echo "AVG_NUM = ${AVG_NUM}"
    echo "LAST_EPOCH = ${LAST_EPOCH}"
    echo "AVG_CHECKPOINT_OUT = ${AVG_CHECKPOINT_OUT}"
    echo "---- Score by averaging last checkpoints ${AVG_NUM} -> ${AVG_CHECKPOINT_OUT}"
    echo "Generating average checkpoints..."
#    exit 1

    if [ -f ${AVG_CHECKPOINT_OUT} ]; then
        echo "File ${AVG_CHECKPOINT_OUT} exists...."
    else
        python ../scripts/average_checkpoints.py \
        --user-dir ${user_dir} \
        --inputs ${TRAIN_DIR} \
        --ema=True \
        --ema_decay=${EMA_DECAY} \
        --num-epoch-checkpoints ${AVG_NUM} \
        --checkpoint-upper-bound ${UPPERBOUND} \
        --output ${AVG_CHECKPOINT_OUT}
        echo "Finish generating averaged, start generating samples"
    fi

    export CHECKPOINT=${AVG_CHECKPOINT_OUT}


else
	echo "INFERMODE invalid: ${INFERMODE}"
	exit 1
fi


echo "Start generating"

if [ ${HEAT} == "y" ]; then
    export HEAT_DIR=${GEN_DIR}/heatmaps
    mkdir -p ${HEAT_DIR}
    eval "python ../generate_heatmap.py ${DATA_DIR} \
        --user-dir ${user_dir} \
        --task ${TASK} \
        --path ${CHECKPOINT} \
        --left-pad-source ${LEFT_PAD_SRC} \
        --max-sentences 1 \
        --beam ${BEAM} \
        --gen-subset ${GENSET} \
        --lenpen ${LENPEN} \
        --score-reference \
        --print-alignment \
        --image-dir ${HEAT_DIR} \
        ${rm_srceos_s} ${rm_lastpunct_s} ${get_entropies_s}"

#        --max-tokens ${INFER_BSZ} \
#        --remove-bpe \
#        --user-dir ${user_dir} \
#            | tee ${GEN_OUT}
else

    if [ ${NEWCODE} == "y" ]; then

#        if [ ${RMBPE} == "y" ]; then
#            eval "$(which fairseq-generate) ${DATA_DIR} \
#            --user-dir ${user_dir} \
#            --path ${CHECKPOINT} \
#            --left-pad-source ${LEFT_PAD_SRC} \
#            --max-tokens ${INFER_BSZ} \
#            --beam ${BEAM} \
#            --gen-subset ${GENSET} \
#            --lenpen ${LENPEN} \
#            ${rm_bpe_s} ${rm_srceos_s} | tee ${GEN_OUT}"
#        #    --batch-size ${INFER_BSZ} \
#
#        else
#            $(which fairseq-generate) ${DATA_DIR} \
#            --user-dir ${user_dir} \
#            --path ${CHECKPOINT} \
#            --left-pad-source ${LEFT_PAD_SRC} \
#            --max-tokens ${INFER_BSZ} \
#            --gen-subset ${GENSET} \
#            --beam ${BEAM} \
#            --lenpen ${LENPEN} \
#            | tee ${GEN_OUT}
#        fi
        if [ ${INFER_TASK} == "mt" ]; then
            echo "START GENERATE MT"
            eval "$(which fairseq-generate) ${DATA_DIR} \
                --task ${TASK} \
                --user-dir ${user_dir} \
                --path ${CHECKPOINT} \
                --left-pad-source ${LEFT_PAD_SRC} \
                --max-tokens ${INFER_BSZ} \
                --beam ${BEAM} \
                --gen-subset ${GENSET} \
                --lenpen ${LENPEN} \
                ${rm_bpe_s} ${rm_srceos_s} ${rm_lastpunct_s} | tee ${GEN_OUT}"
        elif [ ${INFER_TASK} == "lm" ]; then
            echo "START EVAL_LM"
            eval "$(which fairseq-eval-lm) ${DATA_DIR} \
                --user-dir ${user_dir} \
                --task language_modeling \
                --path ${CHECKPOINT} \
                --max-tokens ${INFER_BSZ} \
                --gen-subset ${GENSET} \
            "
        else
            echo "INFER_TASK = ${INFER_TASK}, wrong"
            exit 1
        fi
    else
#        if [ ${RMBPE} == "y" ]; then
#            python generate.py ${DATA_DIR} \
#            --user-dir ${user_dir} \
#            --path ${CHECKPOINT} \
#            --left-pad-source ${LEFT_PAD_SRC} \
#            --max-tokens ${INFER_BSZ} \
#            --beam ${BEAM} \
#            --gen-subset ${GENSET} \
#            --lenpen ${LENPEN} \
#            --remove-bpe \
#            | tee ${GEN_OUT}
#        #    --batch-size ${INFER_BSZ} \
#
#        else
#            python generate.py ${DATA_DIR} \
#            --user-dir ${user_dir} \
#            --path ${CHECKPOINT} \
#            --left-pad-source ${LEFT_PAD_SRC} \
#            --max-tokens ${INFER_BSZ} \
#            --gen-subset ${GENSET} \
#            --beam ${BEAM} \
#            --lenpen ${LENPEN} \
#            | tee ${GEN_OUT}
#        fi
        eval "python generate.py ${DATA_DIR} \
            --task ${TASK} \
            --user-dir ${user_dir} \
            --path ${CHECKPOINT} \
            --left-pad-source ${LEFT_PAD_SRC} \
            --max-tokens ${INFER_BSZ} \
            --beam ${BEAM} \
            --gen-subset ${GENSET} \
            --lenpen ${LENPEN} \
            ${rm_bpe_s} ${rm_srceos_s} ${rm_lastpunct_s} | tee ${GEN_OUT}"
    fi
fi


echo "---- Score by score.py for mode=${INFERMODE}, avg=${AVG_NUM} -----"
echo "decode bleu from model ${AVG_CHECKPOINT_OUT}"
echo "decode bleu from file ${GEN_OUT}"
echo ".............................."

export SRC=${GEN_OUT}.src
export HYPO=${GEN_OUT}.hypo
export REF=${GEN_OUT}.ref
export REF_TW=${GEN_OUT}.ref.tweak

#S-2942	Dieser Ausspruch einer aufgebrachten Sebelius zu einem hinter ihr sitzenden Berater bei der gestrigen Anhörung im Repräsentantenhaus wurde nach einem Streitgespräch mit dem republikanischen Abgeordneten Billy Long aus Missouri aufgenommen . Dabei ging es darum , ob sie zur Teilnahme an Obamacare verpflichtet sein sollte .
#T-2942	An exasperated Sebelius uttered that phrase , caught by a hot mic , to an aide seated behind her at yesterday &apos;s House hearing following a contentious exchange with Rep. Billy Long , R @-@ Mo . , over whether she should be required to enroll in Obamacare .
#H-2942	-1.660185694694519	This statement by an angry Sebelius to an adviser behind her at yesterday &apos;s House hearing was made after a dispute with Republican MP Billy Long of Missouri about whether she should be required to participate in Obamacare .

grep ^S ${GEN_OUT} | cut -f2- > ${SRC}
grep ^T ${GEN_OUT} | cut -f2- > ${REF}
grep ^H ${GEN_OUT} | cut -f3- > ${HYPO}

python convert_reference.py

export SRC_ATAT=${SRC}.atat
export HYPO_ATAT=${HYPO}.atat
export REF_ATAT=${REF}.atat
export REF_TW_ATAT=${REF}.tweak.atat


export BLEU_OUT=${GEN_OUT}.bleu
export BLEU_OUT_ATAT=${GEN_OUT}.atat.bleu
export BLEU_OUT_TW=${GEN_OUT}.tweak.bleu
export BLEU_OUT_TW_ATAT=${GEN_OUT}.tweak.atat.bleu

export BLEU_OUT_T2T=${GEN_OUT}.bleu.t2t

perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${HYPO} > ${HYPO_ATAT}
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${REF} > ${REF_ATAT}
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${REF_TW} > ${REF_TW_ATAT}
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${SRC} > ${SRC_ATAT}


#perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${REF_TW} > ${REF_TW_ATAT}
if [ ${HEAT} == "y" ]; then
    echo "stop here"

else
    if [ ${GETBLEU} == "y" ]; then

        echo "============================================="
        echo "run for t2t-bleu::::"
        echo "export SRC=${SRC}"
        echo "export HYPO=${HYPO}"
        echo "export REF=${REF}"
        echo "export HYPO_ATAT=${HYPO_ATAT}"
        echo "export REF_ATAT=${REF_ATAT}"
        echo "export REF_TW=${REF_TW}"
        echo "export REF_TW_ATAT=${REF_TW_ATAT}"
        echo ""
        echo "t2t-bleu --translation=${HYPO} --reference=${REF}"
        echo ""


        echo "------ Score raw ------------"
    #    python score.py --sys ${HYPO} --ref ${REF} > ${BLEU_OUT}
        $(which fairseq-score) --sys ${HYPO} --ref ${REF} > ${BLEU_OUT}
        cat ${BLEU_OUT}
        echo ""
        echo "============================================="
        echo "------ Score with ATAT ------------"
    #    python score.py --sys ${HYPO_ATAT} --ref ${REF_ATAT} > ${BLEU_OUT_ATAT}
        $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_ATAT} > ${BLEU_OUT_ATAT}
    #    $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT}
        cat ${BLEU_OUT_ATAT}
        echo ""

        echo "============================================="
        echo "------ Score with TWEAK ------------"
    #    python score.py --sys ${HYPO_ATAT} --ref ${REF_ATAT} > ${BLEU_OUT_ATAT}
        $(which fairseq-score) --sys ${HYPO} --ref ${REF_TW} > ${BLEU_OUT_TW}
    #    $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT}
        cat ${BLEU_OUT_TW}
        echo ""

        echo "============================================="
        echo "------ Score with TWEAK + ATA------------"
    #    python score.py --sys ${HYPO_ATAT} --ref ${REF_ATAT} > ${BLEU_OUT_ATAT}
        $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT} > ${BLEU_OUT_TW_ATAT}
    #    $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT}
        cat ${BLEU_OUT_TW_ATAT}
        echo ""
        echo ""

        echo "============================================="
        echo "------ Score with Tensor2tensor------------"
    #    python score.py --sys ${HYPO_ATAT} --ref ${REF_ATAT} > ${BLEU_OUT_ATAT}
    #    $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT} > ${BLEU_OUT_TW_ATAT}
        t2t-bleu --translation=${HYPO} --reference=${REF} > ${BLEU_OUT_T2T}
    #    $(which fairseq-score) --sys ${HYPO_ATAT} --ref ${REF_TW_ATAT}
        cat ${BLEU_OUT_T2T}
        echo ""
        echo ""

    fi

fi

#export GEN_OUT=${INFER_DIR}/${TESTSET}.tok.nobpe.32000.genout.${TGT_LANG}
#export DECODE_FILE=${INFER_DIR}/${TESTSET}.tok.nobpe.32000.infer.${TGT_LANG}
#export REF=${INFER_DIR}/${TESTSET}.tok.nobpe.32000.ref.${TGT_LANG}
#
## actual inference
#export CHECKPOINT=${TRAIN_DIR}/checkpoint_best.pt
#export BEAM="${BEAM:-5}"
#export INFER_BSZ="${INFER_BSZ:-128}"
#export LENPEN="${LENPEN:-0.6}"
#python generate.py ${DATA_DIR} \
#    --path ${CHECKPOINT} \
#    --batch-size ${INFER_BSZ} \
#    --beam ${BEAM} \
#    --lenpen ${LENPEN} \
#    --remove-bpe | tee ${GEN_OUT}


# averaging things

#export AVG_CHECKPOINTS="${AVG_CHECKPOINTS:-5}"
#export AVG_CHECKPOINT_OUT="${AVG_CHECKPOINT_OUT:-$TRAIN_DIR/averaged_model.pt}"
#echo "---- Score by averaging last checkpoints ${AVG_CHECKPOINTS} -> ${AVG_CHECKPOINT_OUT}"
#echo "Generating average checkpoints..."
#python scripts/average_checkpoints.py \
#	--inputs ${TRAIN_DIR} \
#	--num-epoch-checkpoints ${AVG_CHECKPOINTS} \
#	--output ${AVG_CHECKPOINT_OUT}
#
#echo "Finish generating averaged, start bleuing"
#
#export AVG_GEN_OUT=${GEN_OUT}.avg${AVG_CHECKPOINTS}
#export AVG_HYPO=${AVG_GEN_OUT}.hypo
#export AVG_REF=${AVG_GEN_OUT}.ref
#export AVG_BLEU_OUT=${AVG_GEN_OUT}.bleu_out
#
#python generate.py ${DATA_DIR} \
#    --path ${AVG_CHECKPOINT_OUT} \
#    --batch-size ${INFER_BSZ} \
#    --beam ${BEAM} \
#    --lenpen ${LENPEN} \
#    --remove-bpe | tee ${AVG_GEN_OUT}
#
#
## FIXME: report all results
#grep ^H ${GEN_OUT} | cut -f3- > ${DECODE_FILE}
#grep ^T ${GEN_OUT} | cut -f2- > ${REF}
#
#echo "---- Score by score.py before best_checkpoint -----"
#python score.py --sys ${DECODE_FILE} --ref ${REF}
#
#echo "---- Score by score.py after ATAT for best_checkpoint -----"
#export DECODE_ATAT_S=${DECODE_FILE}.scorepy.atat
#export REF_ATAT_S=${REF}.scorepy.atat
#export BLEU_OUT_ATAT_S=${DECODE_ATAT_S}.bleu
#perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${REF} > ${REF_ATAT_S}
#perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${DECODE_FILE} > ${DECODE_ATAT_S}
#python score.py --sys ${DECODE_ATAT_S} --ref ${REF_ATAT_S} > ${BLEU_OUT_ATAT_S}
#cat ${BLEU_OUT_ATAT_S}
#
#
#echo "---- Score by score.py after ATAT on ${AVG_CHECKPOINTS} averaged_checkpoint: ${AVG_CHECKPOINT_OUT}"
#grep ^H ${AVG_GEN_OUT} | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${AVG_HYPO}
#grep ^T ${AVG_GEN_OUT} | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${AVG_REF}
#python score.py --sys ${AVG_HYPO} --ref ${AVG_REF} > ${AVG_BLEU_OUT}
#cat ${AVG_BLEU_OUT}

#export DECODE_REP_UNI=${DECODE_FILE}.repuni
#export DECODE_TOK=${DECODE_FILE}.tok
#
#export REF_ATAT=${REF}.atat
#export DECODE_ATAT=${DECODE_TOK}.atat
#
#export BLEU_OUT=${DECODE_FILE}.bleu_score
#
## Replace unicode.
#perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l ${TGT_LANG}  < ${DECODE_FILE} > ${DECODE_REP_UNI}
#
## Tokenize.
#perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${TGT_LANG} < ${DECODE_REP_UNI} > ${DECODE_TOK}
#
## Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
#perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${REF} > ${REF_ATAT}
#perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < ${DECODE_TOK} > ${DECODE_ATAT}
#
## Get BLEU.
#perl $mosesdecoder/scripts/generic/multi-bleu.perl ${REF_ATAT} < ${DECODE_ATAT} > ${BLEU_OUT}
#

# FIXME: results for baseline
#---- Score by score.py -----
#Namespace(ignore_case=False, order=4, ref='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.ref.de', sys='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.infer.de')
# todo: BLEU4 = 26.02, 57.6/31.7/19.7/12.7 (BP=1.000, ratio=1.019, syslen=64268, reflen=63078)
#---- Score by score.py after ATAT -----
#Namespace(ignore_case=False, order=4, ref='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.ref.de.scorepy.atat', sys='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.infer.de.scorepy.atat')
# todo BLEU4 = 26.57, 58.2/32.3/20.2/13.1 (BP=1.000, ratio=1.016, syslen=65530, reflen=64496)
#---- Score by averaging last checkpoints 5 -> /data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/averaged_model.pt
#Generating average checkpoints...

#---- Score by score.py after ATAT on 5 averaged_checkpoint: /data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/averaged_model.pt
#Namespace(ignore_case=False, order=4, ref='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.genout.de.avg5.ref', sys='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/INFER/newstest2014.tok.nobpe.32000.genout.de.avg5.hypo')
# todo BLEU4 = 26.77, 58.3/32.5/20.4/13.3 (BP=1.000, ratio=1.019, syslen=65711, reflen=64496)

# u25, 27.21 ????

# todo after epoch 33
#9 -0.1015 -0.0925 -0.0763 -0.1798 -0.2820 -0.1758 -0.3711 -0.0642 -0.0162 -0.5392 -0.1659 -0.3123 -0.3171 -0.7089 -0.1435 -0.1105
#| Translated 3003 sentences (86008 tokens) in 151.5s (19.82 sentences/s, 567.63 tokens/s)
#| Generate test with beam=5: BLEU4 = 26.51, 57.8/32.2/20.2/13.2 (BP=1.000, ratio=1.019, syslen=64292, reflen=63078)
#---- Score by score.py after ATAT for mode=avg, avg=5 -----
#Namespace(ignore_case=False, order=4, ref='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/infer/newstest2014.tok.nobpe.32000.genout.de.avg.avg5.ref', sys='/data/nxphi47/projects/tree/train_fairseq_v2/wmt16_en_de_bpe32k/transformer_wmt_en_de-b4096-gpu1-upfre8/infer/newstest2014.tok.nobpe.32000.genout.de.avg.avg5.hypo')
#BLEU4 = 27.11, 58.4/32.8/20.7/13.6 (BP=1.000, ratio=1.016, syslen=65534, reflen=64496)

# TODO: pay less attention
#https://github.com/pytorch/fairseq/blob/master/examples/pay_less_attention_paper/README.md


