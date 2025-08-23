import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
from device import device
from ggdrive import ggdrive
from model import models
from optimizer import optimizers
from loss_fn import loss_fns
import dataset
from pydantic import BaseModel
from eval import eval
from tqdm import tqdm


class TrainConfig(BaseModel):
    model: str
    epochs: int
    optimizer: str
    loss_fn: str
    batch_size: int
    lr: float


def train_one_epoch(
    model: nn.Module,
    cifar100_train_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(cifar100_train_loader, desc="Training one epoch"):
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

    return avg_loss, acc


def train(args=None):
    project = args.project if args else None

    with wandb.init(project=project) as run:
        run.config.update({"device": device})
        config = TrainConfig(**dict(run.config))

        model = models[config.model]()
        model.to(device=device)

        loss_fn = loss_fns[config.loss_fn]
        optimizer = optimizers[config.optimizer](model.parameters(), lr=config.lr)
        cifar100_train_loader = DataLoader(dataset.cifar100_train, batch_size=config.batch_size)

        for epoch in range(1, config.epochs+1):
            train_loss, train_accuracy = train_one_epoch(
                model,
                cifar100_train_loader,
                loss_fn,
                optimizer
            )
            eval_report = eval(model, loss_fn)

            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                **eval_report.model_dump()
            })

        uploaded_url = ggdrive.upload_weight(model)
        run.config.update({"url": uploaded_url})
