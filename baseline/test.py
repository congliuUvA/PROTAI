"""This module is used for testing model saved in model_checkpoints."""
import torch
from tqdm import tqdm
import hydra
from dataset import VoxelsDataset, GaussianFilter
from pathlib import Path
from torch.utils.data import DataLoader
from model.cnn.model import CNN
from utils.log import get_logger

logger = get_logger(__name__)


def test(ckpt_path: Path, test_dataloader: DataLoader, device):
    # load model
    state_dict = torch.load(str(ckpt_path))
    model = CNN(20, 4)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # test
    num_instance = 0
    correct = 0
    with torch.no_grad():
        for voxels_data, labels in tqdm(test_dataloader):
            voxels_data, labels = voxels_data.to(device), labels.to(device)
            preds = model(voxels_data)
            labels_int = torch.where(labels == 1)[-1].cpu()
            preds_int = torch.max(preds.detach(), dim=1)[-1].cpu()
            correct += (preds_int == labels_int).sum()
            num_instance += preds.shape[0]
    return correct / num_instance


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args):
    """test module."""
    # load arguments
    args_model = args.model
    args_data = args.data

    parent_path = Path().cwd()
    hdf5_file_path = parent_path / args_data.hdf5_file_dir
    dataset_split_csv_path = parent_path / args_data.dataset_split_csv

    # transformation of the datasets
    transformation = GaussianFilter(3) if args_model.use_transform else None

    # settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args_model.best_model_ckpt_path

    test_set = VoxelsDataset(
        hdf5_files_path=hdf5_file_path,
        dataset_split_csv_path=dataset_split_csv_path,
        training=False,
        transform=transformation,
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args_model.batch_size,
        shuffle=False,
    )

    # test
    accuracy = test(model_path, test_dataloader, device)
    logger.info(f"Accuracy on test set is {accuracy:.3f}.")




