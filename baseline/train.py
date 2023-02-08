"""Modules for training script."""
import torch
from tqdm import tqdm
import hydra
import wandb
from dataset import VoxelsDataset, GaussianFilter
from pathlib import Path
from torch.utils.data import DataLoader
from model.cnn.model import CNN
import torch.nn as nn
from utils.log import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


def training(
        model,
        epochs,
        train_loader,
        val_loader,
        loss_func,
        optimizer,
        lr_scheduler,
        device,
        wandb_run,
):
    scaler = torch.cuda.amp.GradScaler(enabled=device.type != "cpu")

    best_acc_val = 0
    checkpoint_dir = Path.cwd() / "model_checkpoints"
    checkpoint_dir.mkdir() if not checkpoint_dir.exists() else None
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        model.train()
        acc_train = 0
        for idx, data_train in enumerate(tqdm(train_loader)):
            loss_train, preds, labels = train_step(
                data_train, model, loss_func,
                optimizer, lr_scheduler,
                device, scaler
            )
            wandb_run.log({"loss_train": loss_train})
            #
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
            acc_train_step = (preds_int == labels_int).sum() / preds.shape[0]
            progress_bar.set_postfix(acc=f'{acc_train_step:.3f}')
            acc_train += acc_train_step
        acc_train = acc_train / (idx + 1)
        wandb_run.log({"acc_train": acc_train})

        model.eval()
        progress_bar = tqdm(val_loader)
        acc_val = 0
        for idx, data_val in enumerate(progress_bar):
            loss_val, preds, labels = val_step(
                data_val, model, loss_func, device
            )
            wandb_run.log({"loss_val": loss_val})
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
            acc_val_step = (preds_int == labels_int).sum() / preds.shape[0]
            progress_bar.set_postfix(acc=f'{acc_val_step:.3f}')
            acc_val += acc_val_step
        acc_val = acc_val / (idx + 1)
        wandb_run.log({"acc_val": acc_val})
        update_best_checkpoint(
            acc_val, best_acc_val,
            epoch, checkpoint_dir, model,
            optimizer, lr_scheduler
        )
        model_save_path = checkpoint_dir / f"CNN_{epoch}.pt"
        save_checkpoint(model, optimizer, lr_scheduler, str(model_save_path), epoch)


def train_step(data_train,
               model,
               loss_func,
               optimizer,
               lr_scheduler,
               device,
               scaler):
    voxel_boxes, labels = data_train
    voxel_boxes, labels = voxel_boxes.to(device), labels.to(device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        preds = model(voxel_boxes)  # (bs, 20)
        loss_train = loss_func(preds, labels)
    scaler.scale(loss_train).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()
    return loss_train.item(), preds, labels


def val_step(data_val, model, loss_func, device):
    with torch.no_grad():
        voxel_boxes, labels = data_val
        voxel_boxes, labels = voxel_boxes.to(device), labels.to(device)
        preds = model(voxel_boxes)  # (bs, 20)
        loss_val = loss_func(preds, labels)
    return loss_val.item(), preds, labels


def update_best_checkpoint(acc_val, best_acc_val, epoch, checkpoint_dir, model, optimizer, lr_scheduler,):
    if acc_val > best_acc_val:
        best_acc_val = acc_val
        best_checkpoint_path = checkpoint_dir / f"CNN_{epoch}_best_acc_{acc_val}.pt"
        save_checkpoint(model, optimizer, lr_scheduler, best_checkpoint_path, epoch)
    return best_acc_val, best_checkpoint_path


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    checkpoint_path: str,
    epoch: int,
):
    """Save a model's checkpoint.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer used for training, to save.
        lr_scheduler: PyTorch learning rate scheduler to be called after optimizer's update.
        checkpoint_path: Path (str format) at which to save the checkpoint.
        epoch: Checkpoint's epoch.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(state, checkpoint_path)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig):
    # load arguments of 3DCNN
    args_model = args.model
    args_data = args.data

    parent_path = Path().cwd()
    hdf5_file_path = parent_path / args_data.hdf5_file_dir
    dataset_split_csv_path = parent_path / args_data.dataset_split_csv

    val_set = VoxelsDataset(
        hdf5_files_path=hdf5_file_path,
        dataset_split_csv_path=dataset_split_csv_path,
        val=True,
        transform=GaussianFilter,
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=args_model.batch_size,
        shuffle=False,
    )

    loss_func = nn.CrossEntropyLoss()
    model = CNN(args_model.num_classes, args_model.num_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99**epoch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold in range(args_model.folds):
        # initialize wandb
        wandb_run = wandb.init(
            project="3DCNN",
            config=OmegaConf.to_container(args, resolve=True),
            reinit=True,
        )
        with wandb_run:
            train_set = VoxelsDataset(
                hdf5_files_path=hdf5_file_path,
                dataset_split_csv_path=dataset_split_csv_path,
                training=True,
                fold=fold,
                transform=GaussianFilter
            )
            train_dataloader = DataLoader(
                dataset=train_set,
                batch_size=args_model.batch_size,
                shuffle=True,
            )

            training(
                model, args_model.epochs, train_dataloader,
                val_dataloader, loss_func, optimizer, lr_scheduler,
                device, wandb_run
            )


