import os
import sys
import random
import numpy as np
import argparse

import torch
import torchvision
from torchvision import transforms

from gta import gta
from test import test
from data_loader import (
    amazon_source,
    amazon_target,
    webcam_source,
    webcam_target,
    dslr_source,
    dslr_target,
    classes
)



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
# parser.add_argument('--method', default='GTA', help='Method to train| GTA, sourceonly')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--adv_weight', type=float, default = 0.1, help='weight for adv loss')
parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay, default=0.0002')
parser.add_argument('--alpha', type=float, default = 0.3, help='multiplicative factor for target adv. loss')


opt = parser.parse_args()
print(opt)

# Creating directories 
try:
    os.makedirs(opt.outf)
except OSError:
    pass

try:
    os.makedirs(os.path.join(opt.outf, 'models'))
except OSError:
    pass


random.seed(42)
torch.manual_seed(42)
if opt.gpu >= 0:
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True

loaders_ = {
    "amazon_source": amazon_source,
    "amazon_target": amazon_target,
    "webcam_source": webcam_source,
    "webcam_target": webcam_target,
    "dslr_source": dslr_source,
    "dslr_target": dslr_target,
}

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def main(source_dataset, target_dataset, model_root):
    dataloader_source = loaders_[source_dataset]
    dataloader_target = loaders_[target_dataset]

    source = source_dataset.split("_")[0]
    target = target_dataset.split("_")[0]

    nclasses = len(classes)

    model = gta(opt, nclasses, mean, std, dataloader_source, dataloader_target)
    model.train()

if __name__ == '__main__':
    source_dataset_name = "amazon_source"
    target_dataset_name = "webcam_target"

    main(source_dataset_name, target_dataset_name)
