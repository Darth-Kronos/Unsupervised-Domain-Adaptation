import os
import random

import torch
import torchvision.transforms as transforms
from torchmetrics import F1Score

random.seed(42)
torch.manual_seed(42)


def evaluate(args, dataloader, epoch):
    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    batch_size = 128
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
    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = perturb_loader(data_target)
    
        batch_size = len(t_label)

        if device == "cuda":
            t_img = t_img.to(device)
            t_label = t_label.to(device)

        class_output, _ = net(input=t_img, alpha=alpha)

        pred = torch.argmax(class_output, dim=1)
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        f1_running += f1(pred.cpu(), t_label.data.view_as(pred).cpu())

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    f1_running /= n_total

    return accu, f1_running

def add_perturbations(data):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=55, sigma=(0.1, 5.0)),
        ], p=0.75),
        transforms.ToTensor(),
    ])
    return transform(data)


def perturb_loader(loader):
    data, target = loader
    data_perturbed = torch.stack([add_perturbations(d) for d in data])
    return data_perturbed, target

def run_perturbations(args, writer, dataloader):
    accuracy, f1 = evaluate(args, dataloader, epoch=29)

    writer.add_scalar(f"Perturbations/Accuracy", accuracy, global_step=None)
    writer.add_scalar(f"Perturbations/F1-Score", f1, global_step=None)