#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
export WANDB_RUN_GROUP="multisample-dropout"
export WANDB_NOTES=""
export WANDB_PROJECT="pii-dd"

# python train_v2.py --model_path "microsoft/deberta-v3-large" --max_length 768 --stride 256 --output_dir "dv3l-all-with-datav2" --lr 2e-5  

            # --lora_modules "query_proj,key_proj,value_proj,pos_key_proj,pos_query_proj" \
            # --use_lora \

lr=("8e-6" "1e-5" "3e-5")
layer_drop_p=("0.1")

for i in "${lr[@]}"
do
    for p in "${layer_drop_p[@]}"
    do
        for x in {0..3}
        do
            python strided_train.py --model_path "microsoft/deberta-v3-large" \
            --max_length 512 \
            --stride 128 \
            --output_dir "outputs/d3l_{$i}_{$x}_ld{$p}_msd" \
            --lr $i \
            --fold $x \
            --save_strategy "no" \
            --filter_no_pii_percent_allow 0.3 \
            --num_train_epochs 2 \
            --warmup_steps 100 \
            --add_newline_token True \
            --gradient_checkpointing False \
            --model_dtype fp32 \
            --optim "adamw_8bit" \
            --layer_drop_prob $p \
            --per_device_train_batch_size 6 \
            --per_device_eval_batch_size 6 \
            --use_multisample_dropout \
            --adam_beta2 0.98 \
            --adam_epsilon 1e-6 
    
        done
    done
done


# for i in "${lr[@]}"
# do
#     for x in {1..4}
#     do
#         python train_v2.py --lr $i --model_path "microsoft/deberta-v3-large" --max_length 512 --stride 128 --output_dir "."
#     done
# done
