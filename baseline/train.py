"""Modules for training script."""
import torch
from tqdm import tqdm
import hydra
import wandb
from dataset import VoxelsDataset, GaussianFilter
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from model.cnn.model import CNN
import torch.nn as nn
from utils.log import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


def training(
        model,
        fold,
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
    best_ckpt_path = ""
    checkpoint_dir = Path.cwd() / "model_checkpoints"
    checkpoint_dir.mkdir() if not checkpoint_dir.exists() else None
    lr_step_bool = False

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        model.train()
        acc_train = 0
        for train_idx, data_train in enumerate(progress_bar):
            loss_train, preds, labels, lr_step_bool = train_step(
                data_train, model, loss_func,
                optimizer, lr_scheduler, lr_step_bool,
                device, scaler
            )
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
            acc_train_step = (preds_int == labels_int).sum() / preds.shape[0]
            acc_train += acc_train_step
            progress_bar.set_postfix(acc=f'{acc_train / (train_idx + 1):.3f}')
            wandb_run.log({"loss_train": loss_train, "train_axes": train_idx})
            wandb_run.log({"acc_train": acc_train / (train_idx + 1), "train_axes": train_idx})
            wandb_run.log({"learning_rate": optimizer.param_groups[0]['lr'], "train_axes": train_idx})

            # run a val process every 2000 batches
            if train_idx % 2000 == 1999:
                model.eval()
                progress_bar = tqdm(val_loader)
                acc_val = 0
                for val_idx, data_val in enumerate(progress_bar):
                    preds, labels = val_step(
                        data_val, model, loss_func, device
                    )
                    labels_int = torch.where(labels == 1)[-1].cpu()
                    preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
                    acc_val_step = (preds_int == labels_int).sum() / preds.shape[0]
                    acc_val += acc_val_step
                    progress_bar.set_postfix(acc=f'{acc_val / (val_idx + 1):.3f}')
                    wandb_run.log({"acc_val": acc_val / (val_idx + 1), "val_axes": val_idx})
                prev_acc_val = best_acc_val
                best_acc_val, best_ckpt_path = update_best_checkpoint(
                    acc_val, best_acc_val, best_ckpt_path,
                    fold, epoch, checkpoint_dir, model,
                    optimizer, lr_scheduler
                )
                # if acc on val is decreased or not increased by 0.1%, step learning rate.
                if (best_acc_val - prev_acc_val) / (prev_acc_val + 1e-7) <= 0.001:
                    lr_step_bool = True

        # regularly save model once one epoch is finished.
        model_save_path = checkpoint_dir / f"{fold}_CNN_{epoch}.pt"
        save_checkpoint(model, optimizer, lr_scheduler, str(model_save_path), epoch)

    return best_ckpt_path, best_acc_val


def train_step(data_train,
               model,
               loss_func,
               optimizer,
               lr_scheduler,
               lr_step_bool,
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
    if lr_step_bool:
        lr_scheduler.step()
        lr_step_bool = False
    return loss_train.item(), preds, labels, lr_step_bool


def val_step(data_val, model, loss_func, device):
    with torch.no_grad():
        voxel_boxes, labels = data_val
        voxel_boxes, labels = voxel_boxes.to(device), labels.to(device)
        preds = model(voxel_boxes)  # (bs, 20)
    return preds, labels


def update_best_checkpoint(acc_val, best_acc_val, best_checkpoint_path,
                           fold, epoch, checkpoint_dir, model, optimizer, lr_scheduler,):
    if acc_val > best_acc_val:
        best_acc_val = acc_val
        best_checkpoint_path = checkpoint_dir / f"{fold}_CNN_{epoch}_best_acc_{acc_val}.pt"
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
    hdf5_file_path = Path("/ssdstore/cliu3") / args_data.hdf5_file_dir
    dataset_split_csv_path = parent_path.parent / args_data.dataset_split_csv

    # transformation of the datasets
    transformation = GaussianFilter(3) if args_model.use_transform else None

    val_set = VoxelsDataset(
        hdf5_files_path=hdf5_file_path,
        dataset_split_csv_path=dataset_split_csv_path,
        val=True,
        transform=transformation,
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=3500,
        shuffle=False,
        num_workers=args_model.num_workers,
        pin_memory=True,
    )

    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(args_model.num_classes, args_model.num_channels, args_model.drop_out).to(device)
    model = nn.DataParallel(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args_model.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args_model.learning_rate, momentum=0.75)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    fold_idx = "all"
    # initialize wandb
    wandb_run = wandb.init(
        project="3DCNN",
        config=OmegaConf.to_container(args, resolve=True),
        reinit=True,
    )
    best_ckpt_path_list, best_acc_list = [], []
    with wandb_run:
        train_set = VoxelsDataset(
            hdf5_files_path=hdf5_file_path,
            dataset_split_csv_path=dataset_split_csv_path,
            training=True,
            fold=fold_idx,
            k_fold_test=False,
            transform=transformation,
        )
        weighted_sampler = WeightedRandomSampler(
            weights=train_set.proportion_list,
            num_samples=train_set.length,
            replacement=True,
        )
        sampler = BatchSampler(
            sampler=weighted_sampler,
            batch_size=args_model.batch_size,
            drop_last=False
        )
        train_dataloader = DataLoader(
            sampler=sampler if args_model.use_sampler else None,
            dataset=train_set,
            batch_size=args_model.batch_size,
            shuffle=True,
            num_workers=args_model.num_workers,
            pin_memory=True,
        )

        best_ckpt_path, best_val_acc = training(
            model, fold_idx, args_model.epochs, train_dataloader,
            val_dataloader, loss_func, optimizer, lr_scheduler,
            device, wandb_run
        )

        # record best_accs and corresponding path
        best_ckpt_path_list.append(best_ckpt_path)
        best_acc_list.append(best_val_acc)


if __name__ == "__main__":
    logger.info("start training!")
    main()
