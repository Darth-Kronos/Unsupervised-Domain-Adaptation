import torch

class ReverseLayerF(torch.autograd.Function):
    def forward(self, x, alpha):
        self.alpha = alpha
    
        return x.view_as(x)
    
    def backward(self, grad_outputs):
        output = grad_outputs.neg() * self.alpha
        
        return output, None
    
class DANNModel(torch.nn.Module):
    def __init__(self):
        super(DANNModel, self).__init__()
        
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 50, kernel_size=5),
            torch.nn.BatchNorm2d(50),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
        )
        
        self.class_classifier = torch.nn.Sequential(
            torch.nn.Linear(50 * 53 * 53, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(True),
            torch.nn.Dropout1d(),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 31),
            torch.nn.LogSoftmax(dim=1),
        )
        
        self.domain_classifier = torch.nn.Sequential(
            torch.nn.Linear(50 * 53 * 53, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 3),
            torch.nn.LogSoftmax(dim=1),
        )
        
    def forward(self, input, alpha):
        feature = self.feature(input)
        feature = feature.view(input.data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output