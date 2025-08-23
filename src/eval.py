from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import dataset


class EvaluateReport(BaseModel):
    validation_loss: float
    validation_accuracy: float


def eval(model: nn.Module, loss_fn: nn.Module) -> EvaluateReport:
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    device = next(model.parameters()).device
    cifar100_val_loader = DataLoader(dataset.cifar100_val, batch_size=64)

    with torch.no_grad():
        for images, labels in tqdm(cifar100_val_loader, desc="Evaluating one epoch"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / total
    accuracy = correct / total

    return EvaluateReport(
        validation_loss=avg_loss,
        validation_accuracy=accuracy
    )
