#!/bin/bash

export HF_HUB_CACHE="/drive2/hf-cache"
export HF_DATASETS_CACHE="/drive2/hf-cache"
m1="/drive2/kaggle/pii-dd/piidd/training/url_classifier/d3b-url-v1/checkpoint-55"
m2="/drive2/kaggle/pii-dd/piidd/training/url_classifier/d3b-url-v2/checkpoint-44"
m3="/drive2/kaggle/pii-dd/piidd/training/url_classifier/d3b-url-v3/checkpoint-40"
m4="/drive2/kaggle/pii-dd/piidd/training/url_classifier/d3l-url-v1/checkpoint-190"
m5="/drive2/kaggle/pii-dd/piidd/training/url_classifier/d3l-url-v2/checkpoint-76"

python url_classify.py --model_paths "$m4,$m5" \
 --dataset_path "/drive2/kaggle/pii-dd/data/train.json" \
 --output_path "/drive2/kaggle/pii-dd/piidd/inference/outputs/url_classify/2-d3l.pq"
