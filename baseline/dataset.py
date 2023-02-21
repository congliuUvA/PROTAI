"""Modules for dataset definition."""
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import h5py
from typing import Union, Tuple
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
from tqdm import tqdm


class VoxelsDataset(Dataset):
    """Dataset for loading voxel hdf5 files."""

    def __init__(
            self,
            hdf5_files_path: Path,
            dataset_split_csv_path: Path,
            training: bool = False,
            test: bool = False,
            val: bool = False,
            fold: Union[str, None] = None,
            k_fold_test: bool = False,
            transform=None,
    ):
        """Init.

        Args:
            hdf5_files_path: hdf5 file directory path
            dataset_split_csv_path: dataset_split_csv file path
            training: if the dataset is used for training or not
            test: if the dataset is used for test or not
            val: if the dataset is used for validation or not
            fold: fold index in training set.
            k_fold_test: set to True if the dataset is used as k-fold test set.
            transform: transformation applied to the data set.
        """
        self.len_accumulate_list = None
        self.hdf5_files_path = hdf5_files_path
        self.dataset_split_csv_path = dataset_split_csv_path
        self.training = training
        self.test = test
        self.val = val
        self.dataset_csv = pd.read_csv(self.dataset_split_csv_path)
        self.set_name = "training" if self.training else "test" if self.test else "validation"
        self.dataset_csv = self.dataset_csv.loc[self.dataset_csv["set"] == self.set_name]
        if self.training:
            assert fold is not None
            # take out all the folds except for the leftover fold if the set is not for test, else only taking out
            # the selected fold as a test set in k-fold val.
            self.dataset_csv = self.dataset_csv.loc[self.dataset_csv["fold"] != fold] if not k_fold_test \
                else self.dataset_csv.loc[self.dataset_csv["fold"] == fold]
        self.residue_name = [
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLU", "GLN", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL"
        ]
        self.residue_name_dic = {name: idx for idx, name in enumerate(self.residue_name)}
        self.transform = transform
        self.len_accumulate_list = []
        self.length = 0
        self.updated_csv = self.dataset_csv.copy()
        self.gen_updated_csv()

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of directory.
        """
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Load the data corresponding to the given index.

        Args:
            idx: sample index, value will be 0 to self.len()-1.

        Returns:
            Loaded data and its label
        """
        # given one specific data instance idx, find the corresponding position in
        # accumulated length list, the idx + 1 represents the index of the row in the csv file
        row_idx = np.where(np.array(self.len_accumulate_list) <= idx)[0][0] + 1 \
            if idx >= self.len_accumulate_list[0] else 0
        # use the given idx to minus the last accumulated length to derive the residual index in
        # the row hdf5 file
        residue_idx = idx - self.len_accumulate_list[row_idx - 1] if row_idx > 0 else idx
        row = self.updated_csv.iloc[row_idx]
        hdf5_file = f"{row.id}_pdb1.hdf5"
        chain_id = row.full_id.split("_")[-1]
        f = h5py.File(hdf5_file, "r")
        data = f[chain_id][list(f[chain_id].keys())[residue_idx]]
        label_idx = self.residue_name_dic[str(data.attrs["residue_name"])]
        label = torch.zeros(len(self.residue_name))
        label[label_idx] = 1.0
        voxel = torch.tensor(data[:]) + 0
        # use gaussian fiter
        if self.transform:
            voxel = self.transform(voxel)

        return voxel, label

    def gen_updated_csv(self):
        # length accumulate list, prepared for indexing specific box in the data set.
        delete_row_idx = []
        # iterate through all files in sub set
        for idx, row in enumerate(tqdm(self.dataset_csv.itertuples(), total=len(self.dataset_csv.index))):
            pdb_full_id = row.full_id
            chain = pdb_full_id.split("_")[-1]
            pdb_id = row.id
            hdf5_file = self.hdf5_files_path / f"{pdb_id}_pdb1.hdf5"
            if not hdf5_file.exists():
                delete_row_idx.append(idx)
                continue
            f = h5py.File(hdf5_file, "r")
            if chain not in list(f.keys()) or len(f[chain]) == 0:
                delete_row_idx.append(idx)
                continue
            # each box count as one data sample
            self.length += len(f[chain]["num_boxes_chain"])
            self.len_accumulate_list.append(self.length)
            self.updated_csv.drop(index=[delete_row_idx])


class GaussianFilter(object):
    """Transformation for adding gaussian noise according to different types of atoms."""
    def __init__(self, kernel_size):
        """init module."""
        self.sigma_list = [1.7, 1.45, 1.37, 1.7]  # C, N, O, S van der waals radii
        self.kernel_size = kernel_size

    def __call__(self, img: Tensor) -> Tensor:
        """call module.

        Args:
            img: 3D image to be transformed.

        Returns:
            img: transformed img.
        """
        # use gaussian filter
        for i, channels in enumerate(img):
            img[i] = T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma_list[i])(img[i])

        return img





