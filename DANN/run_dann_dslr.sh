#!/bin/bash
export sources=("amazon_source" "webcam_source" "dslr_source")
export targets=("amazon_target" "webcam_target")
export data_dir="/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/data"
export model_path="/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/DANN/models"

for j in ${targets[@]}; do
    echo "dslr_source and" $j
    CUDA_VISIBLE_DEVICES=8 python main.py \
        --source_dataset "dslr_source" \
        --target_dataset $j \
        --perturb false; \
done
