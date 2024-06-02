#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"

python slm_assertion.py \
  --model_path "/drive2/kaggle/pii-dd/piidd/training/assertion/outputs/2024-04-17/09-19-23/checkpoint-13644" \
  --data_path "/drive2/kaggle/pii-dd/data/corrected_train_v1.json" \
  --df_path "/drive2/kaggle/pii-dd/piidd/inference/outputs/d3l-romulan-seven-836/preds.pq" \
  --output_path "/drive2/kaggle/pii-dd/piidd/inference/outputs/d3l-romulan-seven-836/assertion_outputs.pq"