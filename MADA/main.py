import os
import sys
import random
import numpy as np

import torch
import torchvision
from torchvision import transforms

from mada import MADA
from test import test
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


def main(source_dataset, target_dataset, model_root):
    dataloader_source = loaders_[source_dataset]
    dataloader_target = loaders_[target_dataset]

    source = source_dataset_name.split("_")[0]
    target = target_dataset_name.split("_")[0]

    cuda = False
    torch.backends.cudnn.benchmark = True

    lr = 1e-3
    batch_size = 128
    image_size = 224
    n_epoch = 100
    n_classes = 31
    # load model
    net = MADA(n_classes)

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        net = net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()

            class_output, domain_output = net(input=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain_multi = [
                loss_domain(domain_output[class_idx], domain_label)
                for class_idx in range(n_classes)
            ]

            # training model using target data
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = net(input=t_img, alpha=alpha)
            err_t_domain_multi = [
                loss_domain(domain_output, domain_label)
                for class_idx in range(n_classes)
            ]
            
            err_s_domain = sum(err_s_domain_multi) / n_classes
            err_t_domain = sum(err_t_domain_multi) / n_classes

            loss_d = err_s_domain + err_t_domain

            err = alpha*loss_d + err_s_label
            
            err.backward()
            optimizer.step()

            sys.stdout.write(
                "\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f"
                % (
                    epoch,
                    i + 1,
                    len_dataloader,
                    err_s_label.data.cpu().numpy(),
                    err_s_domain.data.cpu().numpy(),
                    err_t_domain.data.cpu().item(),
                )
            )
            sys.stdout.flush()
            torch.save(
                net,
                "{0}/{1}_{2}_model_epoch_current.pth".format(
                    model_root, source, target
                ),
            )

        print("\n")
        accu_s = test(source_dataset_name)
        print("Accuracy of the %s dataset: %f" % (source, accu_s))
        accu_t = test(target_dataset_name)
        print("Accuracy of the %s dataset: %f\n" % (target, accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(net, f"{model_root}/{source}_{target}_model_epoch_best.pth")

    print("============ Summary ============= \n")
    print("Accuracy of the %s dataset: %f" % (source, best_accu_s))
    print("Accuracy of the %s dataset: %f" % (target, best_accu_t))
    print(
        "Corresponding model was save in "
        + model_root
        + f"/{source}_{target}_model_epoch_best.pth"
    )


if __name__ == "__main__":
    source_dataset_name = "amazon_source"
    target_dataset_name = "webcam_target"
    model_root = (
        "MADA/models"
    )
    main(source_dataset_name, target_dataset_name, model_root)
