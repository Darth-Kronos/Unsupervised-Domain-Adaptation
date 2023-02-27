import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
from model import DANNModel
from torchvision import datasets, transforms
from torchmetrics.classification import F1Score

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


def test(args, dataset_name):
    dataloader = loaders_[dataset_name]
    data_name = dataset_name.split("_")[0]

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    batch_size = 128
    image_size = 224
    alpha = 0

    """ test """
    my_net = torch.load(
        os.path.join(args.model_path, f"{source}_{target}_model_epoch_current.pth")
    )
    my_net = my_net.eval()
    f1 = F1Score(task="multiclass", num_classes=31)

    if device == "cuda":
        my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    f1_running = 0
    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if device == "cuda":
            t_img = t_img.to(device)
            t_label = t_label.to(device)

        class_output, _ = my_net(input=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        
        f1_running += f1(pred, t_label.data.view_as(pred))

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    f1_running /= n_total


    return accu, f1_running
