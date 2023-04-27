import os
import time
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
from mada import MADA
from torchvision import datasets, transforms
from torchmetrics.classification import F1Score
from utils.metrics import update_metrics, log_tensorboard

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


def test(args, dataset_name, metrics, writer, tag, epoch):
    dataloader = loaders_[dataset_name]
    data_name = dataset_name.split("_")[0]

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = True
    batch_size = 128
    image_size = 224
    alpha = 0

    """ test """
    net = torch.load(
        os.path.join(args.model_path, f"{source}_{target}_model_epoch_{epoch}.pth")
    )
    net = net.eval()
    f1 = F1Score(task="multiclass", num_classes=31)

    loss_class = torch.nn.NLLLoss()

    if device == "cuda":
        net = net.to(device)
        loss_class = loss_class.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    f1_running = 0
    times = []
    while i < len_dataloader:
        # test model using target data
        start = time.time()
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)
        if batch_size == 1:
            continue
        
        if device == "cuda":
            t_img = t_img.to(device)
            t_label = t_label.to(device)

        class_output, _ = net(input=t_img, alpha=alpha)
        end = time.time()
        times.append((end-start)/batch_size)
        err_t_label = loss_class(class_output, t_label)

        pred = torch.argmax(class_output, dim=1)

        n_correct += pred.eq(t_label).cpu().sum()
        f1_running += f1(pred.cpu(), t_label.cpu())
        n_total += batch_size

        # update val metrics
        if not metrics is None:
            metrics = update_metrics(metrics, pred, t_label)
            writer.add_scalar(f"Loss/class/{tag}/val", err_t_label, epoch)

        i += 1
        
    if not metrics is None:
        metrics = log_tensorboard(
            writer, f"class/{tag}/val", metrics, epoch, source, target
        )

    accu = n_correct.data.numpy() * 1.0 / n_total
    f1_running /= n_total
    print(f"Inference time: {sum(times) / len(times)}")
    return accu, f1_running, metrics
