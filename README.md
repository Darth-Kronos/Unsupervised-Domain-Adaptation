# Unsupervised-Domain-Adaptation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine learning algorithms and techniques often assume that the training and test sets come from the same source. Furthermore, adequate training of these algorithms relies on large amounts of labeled data which is difficult or expensive to acquire. As such, learning with a limited labeled dataset or deploying data in different domains from the training domain will see a degradation in the performance of the deployed model. Unsupervised domain adaptation (UDA) is a technique that addresses this problem by allowing models to adapt to new domains using only unlabeled data. 

UDA has been studied extensively in recent years, with a variety of methods proposed for computer vision, robotics, natural language processing, and speech recognition. Some popular methods include adversarial training, feature alignment, and self-training. Adversarial training involves training a model to minimize the discrepancy between the source and target domains using a domain classifier. Feature alignment methods aim to align the feature distributions between the source and target domains. 

