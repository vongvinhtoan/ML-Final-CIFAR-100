from configs import settings
import torchvision
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cifar100_train = torchvision.datasets.CIFAR100(root=settings.DATASET_PATH, train=True, transform=transform, download=True)
cifar100_val = torchvision.datasets.CIFAR100(root=settings.DATASET_PATH, train=False, transform=transform, download=True)
