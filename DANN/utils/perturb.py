import torch
from torchvision import transforms

def add_perturbations(data):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(data)


def perturb_loader(loader):
    data, target = loader
    data_perturbed = torch.stack([add_perturbations(d) for d in data])
    return data_perturbed, target