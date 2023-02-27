import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
from mada import MADA
from torchvision import datasets, transforms

from data_loader import (
    amazon_source,
    amazon_target,
    dslr_source,
    dslr_target,
    webcam_source,
    webcam_target,
)

random.seed(42)
torch.manual_seed(42)

loaders_ = {
    "amazon_source": amazon_source,
    "amazon_target": amazon_target,
    "webcam_source": webcam_source,
    "webcam_target": webcam_target,
    "dslr_source": dslr_source,
    "dslr_target": dslr_target,
}


def test(dataset_name):
    dataloader = loaders_[dataset_name]
    data_name = dataset_name.split("_")[0]

    model_root = "models"

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """ test """
    my_net = torch.load(
        os.path.join(model_root, "amazon_webcam_model_epoch_current.pth")
    )
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        
        class_output, _ = my_net(input=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
