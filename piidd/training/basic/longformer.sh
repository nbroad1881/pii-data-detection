#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="4bit-peft"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"

# python train_v2.py --model_path "microsoft/deberta-v3-large" --max_length 768 --stride 256 --output_dir "dv3l-all-with-datav2" --lr 2e-5  


lr=("1e-5" "5e-5" "2e-4")
frac=("0.3")

for i in "${lr[@]}"
do
    for f in "${frac[@]}"
    do
        for x in {1..4}
        do
            python strided_train.py --model_path "allenai/longformer-large-4096" \
            --max_length 2048 \
            --stride 512 \
            --output_dir "lfl_{$i}_{$x}" \
            --lr $i \
            --save_strategy "no" \
            --filter_no_pii_percent_allow $f \
            --num_train_epochs 3 \
            --add_newline_token False \
            --use_lora \
            --gradient_checkpointing False \
            --gradient_accumulation_steps 4 \
            --model_dtype fp32 \
            --lora_modules "query,key,value,query_global,key_global,value_global"
        done
    done
done