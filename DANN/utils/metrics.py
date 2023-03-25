import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Recall

from data_loader import classes


def set_metrics(device, num_classes):
    metrics = {
        "Accuracy": Accuracy(num_classes=num_classes, task="multiclass").to(device),
        "F1-Macro": F1Score(
            num_classes=num_classes, task="multiclass", average="macro"
        ).to(device),
        "Recall": Recall(
            num_classes=num_classes, task="multiclass", average="macro"
        ).to(device),
        "ConfusionMatrix": ConfusionMatrix(
            num_classes=num_classes, task="multiclass", normalize="true"
        ).to(device),
    }
    val_metrics = metrics.copy()
    return metrics, val_metrics


def update_metrics(metrics, outputs, labels):
    metrics["Accuracy"].update(outputs, labels)
    metrics["F1-Macro"].update(outputs, labels)
    metrics["Recall"].update(outputs, labels)
    metrics["ConfusionMatrix"].update(outputs, labels)

    return metrics


def log_tensorboard(writer, tag, metrics, epoch, source, target):
    accuracies = metrics["Accuracy"].compute()

    writer.add_scalar(f"Accuracy/{tag}", accuracies.item(), epoch)
    writer.add_scalar(f"F1-Macro/{tag}", metrics["F1-Macro"].compute(), epoch)
    writer.add_scalar(f"Recall/{tag}", metrics["Recall"].compute(), epoch)

    if "class" in tag:
        names = classes
        if source is not None and target is not None:
            confusion_matrix = metrics["ConfusionMatrix"].compute()

            fig, ax = plt.subplots(figsize=(12, 12))
            plt.rcParams.update({"font.size": 14})
            sns.heatmap(
                confusion_matrix.cpu(),
                annot=False,
                fmt=".2f",
                cmap="Blues",
                ax=ax,
                square=True,
                cbar=False,
                xticklabels=names,
                yticklabels=names,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            # add the image to TensorBoard
            writer.add_figure(f"Confusion Matrix/{tag}", fig, epoch)

    metrics["Accuracy"].reset()
    metrics["F1-Macro"].reset()
    metrics["Recall"].reset()
    metrics["ConfusionMatrix"].reset()

    return metrics
