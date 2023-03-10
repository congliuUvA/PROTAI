"""Modules for training script."""
import torch
from tqdm import tqdm
import hydra
import wandb
from dataset import VoxelsDataset, GaussianFilter
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from model.cnn.model import CNN, ResNet3D, Block
import torch.nn as nn
from utils.log import get_logger
from omegaconf import DictConfig, OmegaConf
import math
import numpy as np
import matplotlib.pyplot as plt

logger = get_logger(__name__)

residue_name = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]


def gen_heat_map(arr: np.ndarray):
    # Create the heatmap plot using Matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    # Customize the plot
    ax.set_xticks(np.arange(len(residue_name)))
    ax.set_yticks(np.arange(len(residue_name)))
    ax.set_xticklabels(residue_name)
    ax.set_yticklabels(residue_name)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add a title to the plot
    ax.set_title("Heatmap")

    return fig


def training(
        args_model,
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
    train_log_idx, val_log_idx = 0, 0
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        model.train()
        acc_train = 0
        loss_train = 0.0
        # confusion matrix for training
        conf_mat_train = np.zeros((args_model.num_classes, args_model.num_classes))
        for idx, data_train in enumerate(progress_bar):
            loss_train_step, preds, labels = train_step(
                data_train, model, loss_func,
                optimizer, lr_scheduler,
                device, scaler
            )
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()

            # accumulate loss and acc.
            acc_train_step = (preds_int == labels_int).sum() / preds.shape[0]
            acc_train += acc_train_step
            loss_train += loss_train_step

            # record confusion matrix
            labels_numpy = labels_int.numpy()
            preds_numpy = preds_int.numpy()
            conf_mat_train[labels_numpy, preds_numpy] += 1
            conf_mat_train = conf_mat_train / np.sum(conf_mat_train, 1).reshape(-1, 1)
            fig = gen_heat_map(conf_mat_train)
            progress_bar.set_postfix(acc=f'{acc_train / (idx + 1):.3f}',
                                     loss=f'{loss_train / (idx + 1):.3f}')
            wandb_run.log({"loss_train": loss_train / (idx + 1), "train_axis": train_log_idx})
            wandb_run.log({"acc_train": acc_train / (idx + 1), "train_axis": train_log_idx})
            wandb_run.log({"loss_train_step": loss_train_step, "train_axis": train_log_idx})
            wandb_run.log({"acc_train_step": acc_train_step, "train_axis": train_log_idx})
            wandb_run.log({"learning_rate": optimizer.param_groups[0]['lr'], "train_axis": train_log_idx})
            wandb_run.log({'confusion_matrix_train': wandb.Image(fig), "train_axis": train_log_idx})
            train_log_idx += 1
            plt.close(fig)

        model.eval()
        progress_bar = tqdm(val_loader)
        acc_val = 0
        loss_val = 0
        # confusion matrix for val
        conf_mat_val = np.zeros((args_model.num_classes, args_model.num_classes))
        for idx, data_val in enumerate(progress_bar):
            preds, labels, loss_val_step = val_step(
                data_val, model, loss_func, device
            )
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
            acc_val_step = (preds_int == labels_int).sum() / preds.shape[0]
            acc_val += acc_val_step
            loss_val += loss_val_step

            # record confusion matrix
            labels_numpy = labels_int.numpy()
            preds_numpy = preds_int.numpy()
            conf_mat_val[labels_numpy, preds_numpy] += 1
            conf_mat_val = conf_mat_val / np.sum(conf_mat_val, 1).reshape(-1, 1)
            fig = gen_heat_map(conf_mat_val)
            progress_bar.set_postfix(acc=f'{acc_val / (idx + 1):.3f}')
            wandb_run.log({"loss_val": loss_val / (idx + 1), "val_axis": val_log_idx})
            wandb_run.log({"acc_val": acc_val / (idx + 1), "val_axis": val_log_idx})
            wandb_run.log({"loss_val_step": loss_val_step, "val_axis": val_log_idx})
            wandb_run.log({"acc_val_step": acc_val_step, "val_axis": val_log_idx})
            wandb_run.log({'confusion_matrix_val': wandb.Image(fig), "val_axis": val_log_idx})
            val_log_idx += 1
            plt.close(fig)
        best_acc_val, best_ckpt_path = update_best_checkpoint(
            args_model, acc_val / (idx + 1), best_acc_val, best_ckpt_path,
            fold, epoch, checkpoint_dir, model,
            optimizer, lr_scheduler
        )

        # regularly save model once one epoch is finished.
        model_save_path = checkpoint_dir / f"{fold}_{args_model.model_name}_{epoch}.pt"
        save_checkpoint(model, optimizer, lr_scheduler, str(model_save_path), epoch)

        lr_scheduler.step()

    return best_ckpt_path, best_acc_val


def train_step(data_train,
               model,
               loss_func,
               optimizer,
               lr_scheduler,
               device,
               scaler):
    voxel_boxes, labels = data_train
    voxel_boxes = torch.where(torch.isnan(voxel_boxes), torch.full_like(voxel_boxes, 0), voxel_boxes)
    voxel_boxes = torch.where(torch.isinf(voxel_boxes), torch.full_like(voxel_boxes, 0), voxel_boxes)
    voxel_boxes, labels = voxel_boxes.to(device), labels.to(device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        preds = model(voxel_boxes)  # (bs, 20)
        loss_train = loss_func(preds, labels)
    scaler.scale(loss_train).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()

    # optimizer.zero_grad()
    # preds = model(voxel_boxes)  # (bs, 20)
    # loss_train = loss_func(preds, labels)
    # loss_train.backward()
    # optimizer.step()
    # lr_scheduler.step()

    return loss_train.item(), preds, labels,


def val_step(data_val, model, loss_func, device):
    with torch.no_grad():
        voxel_boxes, labels = data_val
        voxel_boxes = torch.where(torch.isnan(voxel_boxes), torch.full_like(voxel_boxes, 0), voxel_boxes)
        voxel_boxes = torch.where(torch.isinf(voxel_boxes), torch.full_like(voxel_boxes, 0), voxel_boxes)
        voxel_boxes, labels = voxel_boxes.to(device), labels.to(device)
        preds = model(voxel_boxes)  # (bs, 20)
        loss_val = loss_func(preds, labels)
    return preds, labels, loss_val


def update_best_checkpoint(args_model, acc_val, best_acc_val, best_checkpoint_path,
                           fold, epoch, checkpoint_dir, model, optimizer, lr_scheduler, ):
    if acc_val > best_acc_val:
        best_acc_val = acc_val
        best_checkpoint_path = checkpoint_dir / f"{fold}_{args_model.model_name}_{epoch}_best_acc_{acc_val}.pt"
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


def get_optimizer(args_model, model):
    optimizer, lr_scheduler = None, None
    if args_model.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args_model.lr)
        lr_func = lambda cur_iter: cur_iter / args_model.warm_up_step \
            if cur_iter < args_model.warm_up_step \
            else (args_model.lr_min +
                  0.5 * (args_model.lr_max - args_model.lr_min) *
                  (1.0 + math.cos(
                      (cur_iter - args_model.warm_up_step) / (args_model.lr_period - args_model.warm_up_step) * math.pi)
                   )
                  ) / args_model.lr_max
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    if args_model.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args_model.lr, momentum=0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=args_model.lr)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    return optimizer, lr_scheduler


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig):
    # load arguments of 3DCNN
    args_model = args.model
    args_data = args.data

    parent_path = Path().cwd()
    data_path_dir = Path("/ssdstore/cliu3") if args_data.use_hddstore else parent_path.parent
    hdf5_file_path = data_path_dir / args_data.hdf5_file_dir
    dataset_split_csv_path = parent_path.parent / args_data.dataset_split_csv

    # transformation of the datasets
    transformation = GaussianFilter(3) if args_model.use_transform else None

    val_set = VoxelsDataset(
        hdf5_files_path=hdf5_file_path,
        dataset_split_csv_path=dataset_split_csv_path,
        val=True,
        transform=transformation,
        use_sampler=args_model.use_sampler,
    )
    logger.info(f"validation set has {val_set.length} instances.")
    logger.info(val_set.freq)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=args_model.batch_size,
        shuffle=False,
        num_workers=args_model.num_workers,
        pin_memory=True,
    )

    fold_idx = "all"
    train_set = VoxelsDataset(
        hdf5_files_path=hdf5_file_path,
        dataset_split_csv_path=dataset_split_csv_path,
        training=True,
        fold=fold_idx,
        k_fold_test=False,
        transform=transformation,
        use_sampler=args_model.use_sampler
    )
    logger.info(f"training set has {train_set.length} instances.")

    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args_model.model_name == "CNN":
        model = CNN(args_model.num_classes, args_model.num_channels, args_model.drop_out)
    if args_model.model_name == "ResNet":
        model = ResNet3D(Block, [1, 1, 1, 1], args_model.num_classes, args_model.num_channels)
    model = nn.DataParallel(model)
    model = model.to(device)
    if args_model.best_model_ckpt_path != "":
        model_path = parent_path / args_model.model_ckpt_path / args_model.best_model_ckpt_path
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["state_dict"])

    optimizer, lr_scheduler = get_optimizer(args_model, model)
    # initialize wandb
    wandb_run = wandb.init(
        project="3DCNN",
        config=OmegaConf.to_container(args, resolve=True),
        reinit=True,
    )
    best_ckpt_path_list, best_acc_list = [], []
    with wandb_run:
        if args_model.use_sampler:
            weighted_sampler = WeightedRandomSampler(
                weights=train_set.proportion_list,
                num_samples=int(train_set.length),
                replacement=False,
            )
            train_dataloader = DataLoader(
                sampler=weighted_sampler,
                dataset=train_set,
                batch_size=args_model.batch_size,
                num_workers=args_model.num_workers,
                pin_memory=True,
            )
        else:
            train_dataloader = DataLoader(
                dataset=train_set,
                batch_size=args_model.batch_size,
                shuffle=True,
                num_workers=args_model.num_workers,
                pin_memory=True,
            )

        best_ckpt_path, best_val_acc = training(
            args_model, model, fold_idx, args_model.epochs, train_dataloader,
            val_dataloader, loss_func, optimizer, lr_scheduler,
            device, wandb_run
        )

        # record best_accs and corresponding path
        best_ckpt_path_list.append(best_ckpt_path)
        best_acc_list.append(best_val_acc)


if __name__ == "__main__":
    logger.info("start training!")
    main()
