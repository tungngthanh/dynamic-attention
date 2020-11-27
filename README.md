# differentiable window attention

Source code for ACL 2020 paper "Differentiable Window for Dynamic Local Attention"

This is an unofficial example codes for WMT'14 En-Fr
# Training

```bash
export CUDA_VISIBLE_DEVICES=0
export HPARAMS=mix_globallocal_block_cross_mix_globallocal_block_encoder_wmt_en_fr
./tree_fairseq/runs/train_lgl_model_customize_en_fr.sh
```

# Evaluating

```bash
cd ./tree_fairseq/runs
export CUDA_VISIBLE_DEVICES=2 
export TRAIN_DIR=../../train_fairseq_v2/wmt14_en_fr/lgl_mix_globallocal_block_encoder_wmt_en_fr-b4096-gpu1-upfre8-1fp16-id1 
export LEFT_PAD_SRC=True 
./infer_model_v2_wmt_en_fr.sh
```