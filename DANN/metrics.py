import os
import numpy as np
import torch
from torchmetrics import Accuracy, F1Score, Recall, ConfusionMatrix

import matplotlib.cm as cm
from data_loader import classes

def set_metrics(device, num_classes):
    metrics = {"Accuracy": Accuracy(num_classes=num_classes, task="multiclass").to(device), 
               "F1-Macro": F1Score(num_classes=num_classes,  task="multiclass", average="macro").to(device), 
               "Recall": Recall(num_classes=num_classes, task="multiclass", average="macro").to(device),
               "ConfusionMatrix": ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true").to(device)
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
    writer.add_scalar(f"Recall/{tag}",  metrics["Recall"].compute(), epoch)   
    
    if "domain" in tag:
        names = [source, target]
    else:
        names = classes
    if source is not None and target is not None:
        confusion_matrix = metrics["ConfusionMatrix"].compute()
        cmap = cm.Blues  # set colormap to blues
        image = cmap(confusion_matrix.cpu(), bytes=True)
        image = np.transpose(image, (2, 0, 1))  # change the order of dimensions to (C, H, W)
        #image = np.expand_dims(image, axis=0)  # add a batch dimension

        # add the image to TensorBoard
        writer.add_image(f'Confusion Matrix/{tag}', image, epoch)
        
    
    metrics["Accuracy"].reset()
    metrics["F1-Macro"].reset()
    metrics["Recall"].reset()
    metrics["ConfusionMatrix"].reset()
    
    return metrics