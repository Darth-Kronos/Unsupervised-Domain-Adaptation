import os
import sys
import random
import numpy as np
import pandas as pd
import argparse

import torch
import torchvision
from torchvision import transforms

from model import DANNModel
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


def main(args):
    dataloader_source = loaders_[args.source_dataset]
    dataloader_target = loaders_[args.target_dataset]

    source = args.source_dataset.split("_")[0]
    target = args.target_dataset.split("_")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.backends.cudnn.benchmark = True

    lr = 1e-3
    batch_size = 128
    image_size = 224
    n_epoch = 100

    # load model
    net = DANNModel()

    # setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if device == "cuda":
        net = net.to(device)
        loss_class = loss_class.to(device)
        loss_domain = loss_domain.to(device)

    for p in net.parameters():
        p.requires_grad = True

    # training
    results = []
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

            if device == "cuda":
                s_img = s_img.to(device)
                s_label = s_label.to(device)
                domain_label = domain_label.to(device)

            class_output, domain_output = net(input=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if device == "cuda":
                t_img = t_img.to(device)
                domain_label = domain_label.to(device)

            _, domain_output = net(input=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

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
                net,
                "{0}/{1}_{2}_model_epoch_current.pth".format(
                    args.model_path, source, target
                ),
            )

        print("\n")
        accu_s, f1_s = test(args, args.source_dataset)
        print("Accuracy of the %s dataset: %f" % (source, accu_s))
        accu_t, f1_t = test(args, args.target_dataset)
        print("Accuracy of the %s dataset: %f\n" % (target, accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(net, f"{args.model_path}/{source}_{target}_model_epoch_best.pth")
        results.append([epoch, err_s_label.data.detach().cpu().numpy(), err_s_domain.data.detach().cpu().numpy(), err_t_domain.data.detach().cpu().item(), accu_s, f1_s, accu_t, f1_t])
    print("============ Summary ============= \n")
    print("Accuracy of the %s dataset: %f" % (source, best_accu_s))
    print("Accuracy of the %s dataset: %f" % (target, best_accu_t))
    print(
        "Corresponding model was save in "
        + args.model_path
        + f"/{source}_{target}_model_epoch_best.pth"
    )

    results_pd = pd.DataFrame(results, columns=['Epoch', 'Source Label Error', 'Source Domain Error', 'Target Domain Error', 'Source Accuracy', 'Source F1', 'Target Accuracy', 'Target F1'])
    results_pd.to_csv(f"{source}_{target}_DANN_results.csv")

if __name__ == "__main__":
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-sd", "--source_dataset", help="training data name")
    argParser.add_argument("-td", "--target_dataset", help="testing data name")
    argParser.add_argument("-dir", "--data_dir", help="directory where data is stored")
    argParser.add_argument("-models", "--model_path", help="directory where to store model checkpoints")

    args = argParser.parse_args()
    
    """source_dataset_name = "amazon_source"
    target_dataset_name = "webcam_target"
    model_root = (
        "/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/DANN/models"
    )"""
    main(args)
