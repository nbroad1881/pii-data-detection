#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"


# python strided_preds.py \
#   --data_path "/drive2/kaggle/pii-dd/data/train.json" \
#   --model_path "/drive2/kaggle/pii-dd/training/fewnerd/d3b-few-nerd-v1" \
#   --max_length 256 \
#   --stride 128 \
#   --batch_size 32 \
#   --dataset_output_path "outputs/fewnerd-d3b/ds.pq" \
#   --tokenized_output_path "outputs/fewnerd-d3b/tds.pq" \
#   --preds_output_path "outputs/fewnerd-d3b/preds.npy"

python post_processing.py \
  --data_path "/drive2/kaggle/pii-dd/data/train.json" \
  --model_dir "/drive2/kaggle/pii-dd/training/fewnerd/d3b-few-nerd-v1" \
  --dataset_path "outputs/fewnerd-d3b/ds.pq" \
  --tokenized_ds_path "outputs/fewnerd-d3b/tds.pq" \
  --preds_path "outputs/fewnerd-d3b/preds.npy" \
  --add_repeated_names \
  --output_csv_path "outputs/fewnerd-d3b/preds.csv" \
  --include_token_text