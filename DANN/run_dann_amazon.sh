#!/bin/bash
export sources=("amazon_source" "webcam_source" "dslr_source")
export targets=("webcam_target" "dslr_target")
export data_dir="/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/data"
export model_path="/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/DANN/models"
for j in ${targets[@]}; do
    echo "amazon_source and" $j
    CUDA_VISIBLE_DEVICES=7 python main.py \
        -sd "amazon_source" \
        -td $j \
        -ep 30 \
        -models $model_path; \
done