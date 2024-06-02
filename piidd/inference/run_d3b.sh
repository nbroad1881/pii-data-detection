#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"

# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/outputs/d3l_012_lr_3e-5_ld_0.1_f0/checkpoint-3246"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/outputs/d3l_012_soup"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/outputs/d1l_1e-5_f0/checkpoint-1882"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/outputs/d1b_012_lr_3e-5_ld_0.0_f0/checkpoint-5166"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/outputs/d3b_012_lr_3e-5_ld_0.0_f0/checkpoint-4818"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-02/23-20-24/0/checkpoint-1174"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-02/23-20-24/1/checkpoint-1168"
# model_path="/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-02/23-20-24/3/checkpoint-1182"
# model_path="bigcode/starpii"
model_path="/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-21/10-48-22/0/checkpoint-3598"
model_path="/drive2/kaggle/pii-dd/piidd/training/basic/multirun/2024-04-20/21-44-52/2/checkpoint-4740"

output_dir="d3l-floral-bird-887-mpware"
data_file="/drive2/kaggle/pii-dd/data/mpware_mixtral8x7b_v1.1-no-i-username.json"

# python strided_preds.py \
#   --data_path $data_file \
#   --model_path $model_path \
#   --max_length 512 \
#   --stride 128 \
#   --batch_size 64 \
#   --dataset_output_path "outputs/${output_dir}/ds.pq" \
#   --tokenized_output_path "outputs/${output_dir}/tds.pq" \
#   --preds_output_path "outputs/${output_dir}/preds.npy" \
#   --add_labels 1

python post_processing.py \
  --data_path $data_file \
  --model_dir $model_path \
  --dataset_path "outputs/${output_dir}/ds.pq" \
  --tokenized_ds_path "outputs/${output_dir}/tds.pq" \
  --preds_path "outputs/${output_dir}/preds.npy" \
  --output_csv_path "outputs/${output_dir}/preds.pq" \
  --include_token_text \
  --output_path "outputs/${output_dir}" \
  --return_all_token_scores \
  --save_char_preds_path "outputs/${output_dir}/char_preds.pkl" \
  --thresholds "0.85,0.9,0.95" \
  --add_repeated_names \
  --remove_name_titles \
  --correct_name_student_preds \
  --remove_bad_categories \
  --check_phone_numbers \
  --calculate_f5 0