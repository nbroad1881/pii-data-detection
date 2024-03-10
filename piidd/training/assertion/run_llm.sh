#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="assertion-multiclass"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"


lr=("1e-5" "5e-5" "2e-4")


python train_llm.py \
    --model_path "mistralai/Mistral-7B-v0.1" \
    --dataset_path "/drive2/kaggle/pii-dd/data/assertion_ds_v1.parquet" \
    --max_length 512 \
    --output_dir "mistral-7b/2" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --num_train_epochs 2 \
    --use_lora \
    --gradient_checkpointing True \
    --model_dtype int4 \
    --lora_modules "all-linear" \
    --fp16 \
    --report_to "wandb" \
    --learning_rate 8e-5 \
    --logging_steps 10 \
    --evaluation_strategy "epoch" \
    --eval_steps 2 \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --dataloader_num_workers 1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --optim "paged_adamw_8bit"


python train_llm.py \
    --model_path "mistralai/Mistral-7B-v0.1" \
    --dataset_path "/drive2/kaggle/pii-dd/data/assertion_ds_v1.parquet" \
    --max_length 512 \
    --output_dir "mistral-7b/3" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --num_train_epochs 2 \
    --use_lora \
    --gradient_checkpointing True \
    --model_dtype int4 \
    --lora_modules "all-linear" \
    --fp16 \
    --report_to "wandb" \
    --learning_rate 1e-4 \
    --logging_steps 10 \
    --evaluation_strategy "epoch" \
    --eval_steps 2 \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --dataloader_num_workers 1 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --optim "paged_adamw_8bit"