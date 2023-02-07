"""Modules for dataset definition."""
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import h5py
from typing import Union, Tuple
import numpy as np
import torch
from torch import Tensor


class VoxelsDataset(Dataset):
    """Dataset for loading voxel hdf5 files."""

    def __init__(
            self,
            hdf5_files_path: Path,
            dataset_split_csv_path: Path,
            training: bool = False,
            test: bool = False,
            val: bool = False,
            fold: Union[int, None] = None,
    ):
        """Init.

        Args:
            hdf5_files_path: hdf5 file directory path
            dataset_split_csv_path: dataset_split_csv file path
            training: if the dataset is used for training or not
            test: if the dataset is used for test or not
            val: if the dataset is used for validation or not
            fold: fold index in training set.
        """
        self.hdf5_files_path = hdf5_files_path
        self.dataset_split_csv_path = dataset_split_csv_path
        self.training = training
        self.test = test
        self.val = val
        self.dataset_csv = pd.read_csv(self.dataset_split_csv_path)
        self.set_name = "training" if self.training else "test" if self.test else "eval"
        self.dataset_csv = self.dataset_csv.loc[self.dataset_csv["set"] == self.set_name]
        if self.training:
            assert fold is not None
            self.dataset_csv = self.dataset_csv.loc[self.dataset_csv["fold"] == fold]
        self.residue_name = [
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLU", "GLN", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL"
        ]
        self.residue_name_dic = {name: idx for idx, name in enumerate(self.residue_name)}

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of directory.
        """
        # length accumulate list, prepared for indexing specific box in the data set.
        self.len_accumulate_list = []
        length = 0
        # iterate through all files in sub set
        for idx, row in self.dataset_csv.iterrows():
            pdb_full_id = row["full_id"]
            chain = pdb_full_id.split("_")[-1]
            pdb_id = row["id"]
            hdf5_file = f"{pdb_id}_pdb1.hdf5"
            f = h5py.File(hdf5_file, "r")
            # each box count as one data sample
            length += len(f[chain].keys())
            self.len_accumulate_list.append(length)
        return length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Load the data corresponding to the given index.

        Args:
            idx: sample index, value will be 0 to self.len()-1.

        Returns:
            Loaded data and its label
        """
        # given one specific data instance idx, find the corresponding position in
        # accumulated length list, the idx + 1 represents the index of the row in the csv file
        row_idx = np.where(np.array(self.len_accumulate_list) <= idx)[0][-1] + 1 \
            if idx >= self.len_accumulate_list[0] else 0
        # use the given idx to minus the last accumulated length to derive the residual index in
        # the row hdf5 file
        residue_idx = idx - self.len_accumulate_list[row_idx - 1] if row_idx > 0 else idx
        row = self.dataset_csv.iloc[row_idx]
        hdf5_file = f"{row.id}_pdb1.hdf5"
        chain_id = row.full_id.split("_")[-1]
        f = h5py.File(hdf5_file, "r")
        data = f[chain_id][list(f[chain_id].keys())[residue_idx]]
        label_idx = self.residue_name_dic[str(data.attrs["residue_name"])]
        label = torch.zeros(len(self.residue_name))
        label[label_idx] = 1.0
        voxel = (torch.tensor(data[:]) + 0).type(torch.FloatTensor)
        return voxel, label



