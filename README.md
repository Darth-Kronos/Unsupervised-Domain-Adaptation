# Unsupervised-Domain-Adaptation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine learning algorithms and techniques often assume that the training and test sets come from the same source. Furthermore, adequate training of these algorithms relies on large amounts of labeled data which is difficult or expensive to acquire. As such, learning with a limited labeled dataset or deploying data in different domains from the training domain will see a degradation in the performance of the deployed model. Unsupervised domain adaptation (UDA) is a technique that addresses this problem by allowing models to adapt to new domains using only unlabeled data. 

UDA has been studied extensively in recent years, with a variety of methods proposed for computer vision, robotics, natural language processing, and speech recognition. Some popular methods include adversarial training, feature alignment, and self-training. Adversarial training involves training a model to minimize the discrepancy between the source and target domains using a domain classifier. Feature alignment methods aim to align the feature distributions between the source and target domains. 

## Domain-Adversarial Training Neural Network
Adapted from [DANN-github](https://github.com/fungtion/DANN) as an implemetnatation of [Domain-Adversarial Training Neural Network](https://arxiv.org/abs/1505.07818)

To train DANN with a source of Amazon for target Webcam a sample command line call is provided

```bash
python main.py --source_dataset "amazon_source" --target_dataset "webcam_target" --model_path "models" --data_dir "data"
```

* must call from within the DANN directory
* last epoch model will save in ```args.model_path```
* TensorBoard outputs will save in ```args.tensorboard_log_dir```
* setting ```args.perturb = True``` will run a perturbed inference after training

## Multi-Adversarial Domain Adaptation 
Implementation of [Multi-Adversarial Domain Adaptation](https://arxiv.org/abs/1809.02176)

To train MADA with a source of Amazon for target Webcam a sample command line call is provided

```bash
python main.py --source_dataset "amazon_source" --target_dataset "webcam_target" --model_path "models" --data_dir "data"
```

* must call from within the MADA directory
* last epoch model will save in ```args.model_path```
* TensorBoard outputs will save in ```args.tensorboard_log_dir```
* setting ```args.perturb = True``` will run a perturbed inference after training

## Contrastive Adaptation Network
Implementation of [Contrastive Adaptation Network](https://arxiv.org/pdf/1901.00976.pdf)
Adopted from [CAN](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)

* Training command
```bash
./experiments/scripts/train.sh ${config_yaml} ${gpu_ids} ${adaptation_method} ${experiment_name}
```
* To run without the model without perturbation, comment line 137 in CAN/tools/test.py

* The experiment log file and the saved checkpoints will be stored at ./experiments/ckpt/${experiment_name}


* Testing command
```bash
./experiments/scripts/train.sh ${config_yaml} ${gpu_ids} ${adaptation_method} ${experiment_name}
```

## CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation
Adopted from [CDTrans](https://github.com/CDTrans/CDTrans) as an implementation of [CDTrans paper](https://arxiv.org/abs/2109.06165).


* Training: We utilize 1 GPU for pre-training and 2 GPUs for UDA, each with 16G of memory.

* Scripts: Command input paradigm

`bash scripts/[pretrain/uda]/office31/run_*.sh [deit_base/deit_small]`

* For example: DeiT-Base scripts
```bash

# Office-31     Source: Amazon   ->  Target: Dslr, Webcam
bash scripts/pretrain/office31/run_office_amazon.sh deit_base
bash scripts/uda/office31/run_office_amazon.sh deit_base

```
* Evaluation

```bash

# Office-31     Source: DSLR   ->  Target: Amazon
python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office/dslr2amazon/t
ransformer_best_model.pth')" DATASETS.NAMES 'Office' DATASETS.NAMES2 'Office' OUTPUT_DIR '../logs/uda/deit_base/office/' DATASETS.ROOT_TRAIN_DIR './data/office31/dslr_list.txt' DATASETS.ROOT_TRAIN_DIR2 './data/office31/dslr_list.txt' DATASETS.ROOT_TEST_DIR './data/office31/amazon_list.txt' PERTURB 'False'

```
