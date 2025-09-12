from configs import settings
import torchvision
import torch
from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
from torch.utils.data import ConcatDataset

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

autoaugment = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

randaugment = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

advprop_aa = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

noisystudent_ra = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


cifar100_transform: dict = {
    "base": base_transform,
    "aa": autoaugment,
    "ra": randaugment,
    "advprop+aa": advprop_aa,
    "noisystudent+ra": noisystudent_ra
}


def get_train_loader(augmentation: str = "base", partition: str = "train"):
    train_set = torchvision.datasets.CIFAR100(
        root=settings.DATASET_PATH, 
        train=True, 
        transform=cifar100_transform[augmentation], 
        download=True
    )
    test_set = torchvision.datasets.CIFAR100(
        root=settings.DATASET_PATH, 
        train=False, 
        transform=cifar100_transform[augmentation], 
        download=True
    )

    if partition == "train":
        return train_set
    elif partition == "test":
        return test_set
    else:
        return ConcatDataset([train_set, test_set])

def get_val_loader():
    return torchvision.datasets.CIFAR100(
        root=settings.DATASET_PATH, 
        train=False, 
        transform=base_transform, 
        download=True
    )
