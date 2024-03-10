#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="url_classifier"

# python train_v2.py --model_path "microsoft/deberta-v3-large" --max_length 768 --stride 256 --output_dir "dv3l-all-with-datav2" --lr 2e-5  


# lr=("3e-5" "4e-5" "5e-5")

# for l in "${lr[@]}"
# do
#     for x in {1..4}
#     do
#         python train.py --model_path "microsoft/deberta-v3-base" \
#         --output_dir "d3b" \
#         --learning_rate $l \
#         --save_strategy "no" \
#         --num_train_epochs 9 \
#         --dataset_path "/drive2/kaggle/pii-dd/data/url_classification_v1.parquet" \
#         --fp16 \
#         --dataloader_num_workers 1 \
#         --weight_decay 0.01 \
#         --per_device_train_batch_size 32 \
#         --per_device_eval_batch_size 32 \
#         --report_to "wandb" \
#         --evaluation_strategy "epoch" \
#         --logging_steps 2 \
#         --lr_scheduler_type "cosine"
#     done

# done

python train.py --model_path "microsoft/deberta-v3-large" \
--output_dir "d3l-only-url-v1" \
--learning_rate 2e-5 \
--save_strategy "epoch" \
--save_total_limit 1 \
--num_train_epochs 5 \
--dataset_path "/drive2/kaggle/pii-dd/data/url_classification_only_url_v1.parquet" \
--fp16 \
--dataloader_num_workers 1 \
--weight_decay 0.01 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--report_to "wandb" \
--evaluation_strategy "epoch" \
--logging_steps 2 \
--lr_scheduler_type "cosine" \
--train_on_all_data False \
--metric_for_best_model "f1" \
--greater_is_better True