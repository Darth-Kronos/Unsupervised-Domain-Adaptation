import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch

from torchvision.models import resnet50, alexnet


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(self, x, alpha):
        self.alpha = alpha
    
        return x.view_as(x)
    
    @staticmethod
    def backward(self, grad_outputs):
        output = grad_outputs.neg() * self.alpha
        
        return output, None

class DomainClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1),
            # torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)
    
class MADA(torch.nn.Module):
    def __init__(self, n_classes):
        super(MADA, self).__init__()
        
        self.n_classes = n_classes

        """self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 50, kernel_size=5),
            torch.nn.BatchNorm2d(50),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
        )"""
        
        self.feature = resnet50(pretrained=True)
        self.feature = torch.nn.Sequential(*(list(self.feature.children())[:-1]))
        

        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024,0.8),
            torch.nn.LeakyReLU(inplace=True),
            #torch.nn.Dropout1d(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(512, 31),
            torch.nn.LogSoftmax(dim=1),
        )
        
        self.domain_classifier_multi = [
            DomainClassifier().cuda() for _ in range(self.n_classes)
        ]
        
        
    def forward(self, input, alpha):
        feature = self.feature(input)
        feature = feature.view(input.data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        
        class_output = self.class_classifier(feature)
        # domain_output = self.domain_classifier(reverse_feature)
        domain_output = []
        for class_idx in range(self.n_classes):
            weighted_reversal_feature = class_output[:, class_idx].unsqueeze(1)*reverse_feature
            domain_output.append(
                self.domain_classifier_multi[class_idx](weighted_reversal_feature)
            )
        
        return class_output, domain_output