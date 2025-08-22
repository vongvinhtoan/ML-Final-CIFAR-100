from configs import settings
import torchvision
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),  # CIFAR-100 mean
                         (0.2673, 0.2564, 0.2762))  # CIFAR-100 std
])

cifar100_train = torchvision.datasets.CIFAR100(root=settings.DATASET_PATH, train=True, transform=transform, download=True)
cifar100_val = torchvision.datasets.CIFAR100(root=settings.DATASET_PATH, train=False, transform=transform, download=True)
