"""Modules for dataset definition."""
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import h5py
from typing import Union, Tuple
import torch
from torch import Tensor
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np


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
            use_sampler: bool = False,
            limit_th: int = 8e6,
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
        self.proportion_list = None
        self.freq = None
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
            if fold != "all":
                # take out all the folds except for the leftover fold if the set is not for test, else only taking out
                # the selected fold as a test set in k-fold val.
                self.dataset_csv = self.dataset_csv.loc[self.dataset_csv["fold"] != fold] if not k_fold_test else \
                    self.dataset_csv.loc[self.dataset_csv["fold"] == fold]
        self.use_sampler = use_sampler
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
        self.look_up_table = {}
        # shuffle the data set to make sure the distribution is somewhat similar
        self.updated_csv = self.dataset_csv.copy().sample(frac=1).reset_index()
        self.limit_th = limit_th
        self.gen_updated_csv()
        # if dataset is used for training, generate weight list for each of the instance,
        # WeightedSampler will sample instances based on their weights, thus yielding batches with similar distribution.
        # if self.training and self.use_sampler:
        #     self.gen_proportion_list()
        self.gen_proportion_list()

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
        info = self.look_up_table[idx]
        hdf5_file, chain_id, residue_idx = info.split("$")[0], info.split("$")[1], info.split("$")[2]
        with h5py.File(hdf5_file, "r") as f:
            data = f[chain_id][residue_idx]
            label_idx = self.residue_name_dic[str(data.attrs["residue_name"])]
            label = torch.zeros(len(self.residue_name))
            label[label_idx] = 1.0
            voxel = (torch.tensor(data[:]) + 0).type(torch.FloatTensor)

        return voxel, label

    def gen_updated_csv(self):
        # length accumulate list, prepared for indexing specific box in the data set.
        data_idx = 0
        # iterate through all files in sub set
        for idx, row in enumerate(tqdm(self.dataset_csv.itertuples(), total=len(self.dataset_csv.index))):
            pdb_full_id = row.full_id
            chain = pdb_full_id.split("_")[-1]
            pdb_id = row.id
            hdf5_file = self.hdf5_files_path / f"{pdb_id}_pdb1.hdf5"
            if not hdf5_file.exists():
                continue
            f = h5py.File(hdf5_file, "r")
            if chain not in list(f.keys()):
                continue
            # each box count as one data sample
            for box_idx in f[chain]:
                label = "" if not self.use_sampler else f[chain][box_idx].attrs["residue_name"]
                self.look_up_table[data_idx] = str(hdf5_file) + "$" + str(chain) + "$" + box_idx + "$" + label
                data_idx += 1
            f.close()
            if data_idx > self.limit_th:
                break
        self.length = data_idx

    def gen_proportion_list(self):
        self.freq = {}
        self.proportion_list = []
        for idx, value in self.look_up_table.items():
            if value.split("$")[-1] not in self.freq:
                self.freq[value.split("$")[-1]] = 1
            else:
                self.freq[value.split("$")[-1]] += 1

        for idx, value in self.look_up_table.items():
            self.proportion_list.append(self.freq[value.split("$")[-1]])

        self.proportion_list = np.array(self.proportion_list) / sum(self.proportion_list)

        # normalizing frequency
        total = sum(self.freq.values())
        for aa, freq in self.freq.items():
            self.freq[aa] = freq / total

        return self.proportion_list


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
