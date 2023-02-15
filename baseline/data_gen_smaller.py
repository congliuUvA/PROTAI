"""This module is for generating a smaller dataset 60000, 10000, 10000 for train/test/val dataset."""

import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import numpy as np


@hydra.main(version_base=None, config_path="../config", config_name="config")
def data_gen_smaller(args: DictConfig):
    """This function generates hdf5 file for all pdb files"""
    # configuration assignment
    args_data = args.data
    args_voxel_box = args.voxel_box
    baseline_dir = Path.cwd()  # baseline/
    root_dir = baseline_dir.parent if not args_data.use_hddtore else "/hddstore/cliu3"

    # dataset split csv path
    dataset_split_csv = root_dir / Path(args_data.dataset_split_csv)

    # dataset csv
    dataset_csv = pd.read_csv(str(dataset_split_csv))

    # create a directory for hdf5 files
    hdf5_file_dir = root_dir / Path(args_data.hdf5_file_dir)

    # create a directory for new hdf5_files
    smaller_hdf5_file_dir = root_dir / Path(args_data.smaller_hdf5_file_dir)
    smaller_hdf5_file_dir.mkdir() if not smaller_hdf5_file_dir.exists() else None

    train_pdb_id = np.unique(np.array(dataset_csv.loc[dataset_csv["set"] == "training"].id))
    test_pdb_id = np.unique(np.array(dataset_csv.loc[dataset_csv["set"] == "test"].id))
    val_pdb_id = np.unique(np.array(dataset_csv.loc[dataset_csv["set"] == "validation"].id))

    num_train, num_test, num_val = 0, 0, 0
    num_train_th, num_test_th, num_val_th = 200, 100, 100

    for pdb_hdf5 in hdf5_file_dir.rglob("*_pdb1.hdf5"):
        pure_pdb_id = pdb_hdf5.name.split(".")[0].split("_")[0]
        if pure_pdb_id in train_pdb_id and num_train < num_train_th:
            os.system(f"cp {pdb_hdf5} {smaller_hdf5_file_dir / pdb_hdf5.name}")
            num_train += 1
        if pure_pdb_id in test_pdb_id and num_test < num_test_th:
            os.system(f"cp {pdb_hdf5} {smaller_hdf5_file_dir / pdb_hdf5.name}")
            num_test += 1
        if pure_pdb_id in val_pdb_id and num_val < num_val_th:
            os.system(f"cp {pdb_hdf5} {smaller_hdf5_file_dir / pdb_hdf5.name}")
            num_val += 1


if __name__ == "__main__":
    data_gen_smaller()


