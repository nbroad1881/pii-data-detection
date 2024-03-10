#!/bin/bash


export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_PROJECT="huggingface"
export WANDB_RUN_GROUP="fewnerd"


python run_ner.py \
  --model_name_or_path "microsoft/deberta-v3-large" \
  --train_file  "/drive2/kaggle/pii-dd/data/fewnerd/persons_grouped_into_100.parquet" \
  --report_to "wandb" \
  --text_column_name "tokens" \
  --label_column_name "ner_tags" \
  --max_seq_length 256 \
  --fp16 \
  --dataloader_num_workers 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --output_dir "d3l-few-nerd-v1" \
  --do_train \
  --do_eval \
  --evaluation_strategy "epoch" \
  --logging_steps 30 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --metric_for_best_model "eval_recall" \
  --greater_is_better True \
  --learning_rate 2e-5 \
  --warmup_steps 100 \
  --overwrite_output_dir