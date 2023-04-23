import os
import sys
import random
import numpy as np
import argparse
import psutil

import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils.perturb import run_perturbations
from utils.parse_args import parse_args
from utils.metrics import set_metrics, update_metrics, log_tensorboard
from mada import MADA
from utils.test import test
from data_loader import (
    amazon_source,
    amazon_target,
    webcam_source,
    webcam_target,
    dslr_source,
    dslr_target,
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
def lr_schedule_step(optimizers, p):
    for param_group in optimizers.param_groups:
        # Update Learning rate
        param_group["lr"] = 1 * 0.01 / (
                    1 + 10 * p) ** 0.75
        # Update weight_decay
        param_group["weight_decay"] = 2.5e-5 * 2

def main(args):
    dataloader_source = loaders_[args.source_dataset]
    dataloader_target = loaders_[args.target_dataset]

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    lr = 0.01
    batch_size = 256
    image_size = 224
    n_epoch = 20
    n_classes = 31
    # load model
    net = MADA(n_classes)

    # setup optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=2.5e-5,
            nesterov=True)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.BCEWithLogitsLoss()

    if device == "cuda":
        net = net.to(device)
        loss_class = loss_class.to(device)
        loss_domain = loss_domain.to(device)

    for p in net.parameters():
        p.requires_grad = True

    log_dir = f"{args.tensorboard_log_dir}/{source}_{target}_{args.encoder}"
    writer = SummaryWriter(log_dir=log_dir)  # create a TensorBoard writer
    c_s_train_metrics, c_s_val_metrics = set_metrics(device, num_classes=31)
    c_t_train_metrics, c_t_val_metrics = set_metrics(device, num_classes=31)

    # training
    best_accu_t = 0.0
    for epoch in range(args.epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            
            lr_schedule_step(optimizer, p)

            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).float()

            if device == "cuda":
                s_img = s_img.to(device)
                s_label = s_label.to(device)
                domain_label = domain_label.to(device)

            class_output, domain_output = net(input=s_img, alpha=alpha)
            class_s_pred = torch.argmax(class_output, dim=1)
            err_s_label = loss_class(class_output, s_label)
            # print(len(domain_output), torch.squeeze(domain_output[0],1).shape, domain_label.shape)
            err_s_domain_multi = [
                loss_domain(torch.squeeze(domain_output[class_idx],1), domain_label)
                for class_idx in range(n_classes)
            ]

            # training model using target data
            data_target = next(data_target_iter)
            t_img, t_label = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).float()

            if device == "cuda":
                t_img = t_img.to(device)
                t_label = t_label.to(device)
                domain_label = domain_label.to(device)

            class_t_output, domain_output = net(input=t_img, alpha=alpha)
            class_t_pred = torch.argmax(class_t_output, dim=1)
            err_t_domain_multi = [
                loss_domain(torch.squeeze(domain_output[class_idx],1), domain_label)
                for class_idx in range(n_classes)
            ]
            err_t_label = loss_class(class_t_output, t_label)

            
            err_s_domain = sum(err_s_domain_multi) / n_classes
            err_t_domain = sum(err_t_domain_multi) / n_classes

            loss_d = err_s_domain + err_t_domain

            err = alpha*loss_d + err_s_label
            
            err.backward()
            optimizer.step()

            # update metrics
            c_s_train_metrics = update_metrics(c_s_train_metrics, class_s_pred, s_label)
            c_t_train_metrics = update_metrics(c_t_train_metrics, class_t_pred, t_label)
            writer.add_scalar(f"Loss/domain/target/train", err_t_domain, epoch)
            writer.add_scalar(f"Loss/class/target/train", err_t_label, epoch)
            writer.add_scalar(f"Loss/class/source/train", err_s_label, epoch)
            writer.add_scalar(f"Loss/overall/train", err, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

            
            sys.stdout.write(
                "\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f"
                % (
                    epoch,
                    i + 1,
                    len_dataloader,
                    err_s_label.data.detach().cpu().numpy(),
                    err_s_domain.data.detach().cpu().numpy(),
                    err_t_domain.data.detach().cpu().item(),
                )
            )
            sys.stdout.flush()
            torch.save(
                net,f"{args.model_path}/{source}_{target}_model_epoch_{epoch}.pth",
            )

        c_s_train_metrics = log_tensorboard(
            writer, "class/source/train", c_s_train_metrics, epoch, source, target
        )
        c_t_train_metrics = log_tensorboard(
            writer, "class/target/train", c_t_train_metrics, epoch, source, target
        )

        # Log RAM usage
        mem_info = psutil.virtual_memory()
        ram_usage = mem_info.used / (1024**3)  # RAM usage in GB
        writer.add_scalar("RAM usage", ram_usage, epoch)

        print("\n")
        accu_s, f1_s, c_s_val_metrics = test(
            args, args.source_dataset, c_s_val_metrics, writer, "source", epoch
        )
        print("Accuracy of the %s dataset: %f" % (source, accu_s))
        accu_t, f1_t, c_t_val_metrics = test(
            args, args.target_dataset, c_t_val_metrics, writer, "target", epoch
        )
        print("Accuracy of the %s dataset: %f\n" % (target, accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(net, f"{args.model_path}/{source}_{target}_model_epoch_best.pth")

        
        if epoch != (args.epochs-1):
            os.remove(f"{args.model_path}/{source}_{target}_model_epoch_{epoch}.pth")

    print("============ Summary ============= \n")
    print("Accuracy of the %s dataset: %f" % (source, best_accu_s))
    print("Accuracy of the %s dataset: %f" % (target, best_accu_t))
    print(
        "Corresponding model was save in "
        + args.model_path
        + f"/{source}_{target}_model_epoch_best.pth"
    )

    if args.perturb:
        print("==== Perturbing Target Dataloader ==== \n")
        run_perturbations(args, writer, dataloader=dataloader_target)
    
    writer.close()


if __name__ == "__main__":
    args = parse_args()

    main(args)
