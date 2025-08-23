import torch
from torch.utils.data import DataLoader
import wandb
from device import device
from model import models
from optimizer import optimizers
from loss_fn import loss_fns
import dataset
from pydantic import BaseModel
from eval import eval


class TrainConfig(BaseModel):
    model: str
    epochs: int
    optimizer: str
    loss_fn: str
    batch_size: int
    lr: float


class TrainReport(BaseModel):
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def train(run: wandb.Run):
    config = TrainConfig(**dict(run.config))

    model = models[config.model]()
    model.to(device=device)

    loss_fn = loss_fns[config.loss_fn]
    optimizer = optimizers[config.optimizer](model.parameters(), lr=config.lr)
    cifar100_train_loader = DataLoader(dataset.cifar100_train, batch_size=config.batch_size)

    for epoch in range(1, config.epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in cifar100_train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        eval_report = eval(model, loss_fn)

        run.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_accuracy": acc,
            **eval_report.model_dump()
        })

    