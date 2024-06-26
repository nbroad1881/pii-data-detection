#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="d1b"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"

# python train_v2.py --model_path "microsoft/deberta-v3-large" --max_length 768 --stride 256 --output_dir "dv3l-all-with-datav2" --lr 2e-5  

            # --lora_modules "query_proj,key_proj,value_proj,pos_key_proj,pos_query_proj" \
            # --use_lora \

lr=("3e-5")
layer_drop_p=("0.0")

for i in "${lr[@]}"
do
    for p in "${layer_drop_p[@]}"
    do
        python strided_train.py --model_path "microsoft/deberta-v3-base" \
        --max_length 512 \
        --stride 128 \
        --output_dir "outputs/d3b_012_lr_${i}_ld_${p}_f0" \
        --lr $i \
        --fold 0 \
        --save_strategy "epoch" \
        --filter_no_pii_percent_allow 0.3 \
        --num_train_epochs 2 \
        --warmup_steps 100 \
        --add_newline_token True \
        --gradient_checkpointing False \
        --model_dtype fp32 \
        --optim "adamw_8bit" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --main_dataset_path "/drive2/kaggle/pii-dd/data/train_012.json" \
        --extra_dataset_path "/drive2/kaggle/pii-dd/data/mixtral-v1a.json"
    done
done
