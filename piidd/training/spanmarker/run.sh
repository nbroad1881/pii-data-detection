#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="spanmarker-names"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"


lr=("2e-5" "8e-5" "3e-4")
frac=("0.3")

# for i in "${lr[@]}"
# do
#     for f in "${frac[@]}"
#     do
#         for x in {1..4}
#         do
#             python strided_train.py --model_path "microsoft/deberta-v2-xlarge" \
#             --max_length 512 \
#             --stride 128 \
#             --output_dir "d2xl" \
#             --lr $i \
#             --save_strategy "no" \
#             --filter_no_pii_percent_allow $f \
#             --num_train_epochs 2 \
#             --add_newline_token False \
#             --use_lora \
#             --gradient_checkpointing False \
#             --model_dtype fp32 \
#             --max_grad_norm 0.3
#         done
#     done
# done


python train.py --model_path "bert-base-cased" \
            --max_length 150 \
            --output_dir "rb-spanmarker-v1" \
            --learning_rate 2e-5 \
            --save_strategy "no" \
            --filter_no_pii_percent_allow 0.1 \
            --num_train_epochs 2 \
            --model_dtype fp32 \
            --report_to "wandb" \
            --weight_decay 0.01 \
            --dataloader_num_workers 1 \
            --fp16 \
            --evaluation_strategy "epoch" \
            --logging_steps 200 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32

