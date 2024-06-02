#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="names"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"


lr=("3e-5")
layer_drop_p=("0")

for i in "${lr[@]}"
do
    for p in "${layer_drop_p[@]}"
    do
        python train.py --model_path "microsoft/deberta-v3-base" \
        --max_length 512 \
        --output_dir "outputs/d3b_lr_${i}" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --num_train_epochs 1 \
        --warmup_steps 100 \
        --add_newline_token False \
        --gradient_checkpointing False \
        --model_dtype fp32 \
        --optim "adamw_8bit" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --dataset_path "/drive2/kaggle/pii-dd/data/name_paragraphs-v2.pq" \
        --evaluation_strategy "epoch" \
        --logging_steps 20 \
        --learning_rate $i \
        --fp16 \
        --dataloader_num_workers 1 \
        --metric_for_best_model "accuracy" \
        --greater_is_better True
    done
done
