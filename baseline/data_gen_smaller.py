"""This module is for generating a smaller dataset 60000, 10000, 10000 for train/test/val dataset.

Only run this module after running deduplicate.py
"""

import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils import log
import h5py
import ray

logger = log.get_logger(__name__)


@ray.remote
def copy_data_instance(hdf5_file_dir, smaller_hdf5_file_dir, pdb_full_id, pdb_id):
    num_copied_data = 0
    # always extract pdb1 because the largest file is pdb1.
    hdf5_file_whole_set = hdf5_file_dir / f"{pdb_id}_pdb1.hdf5"
    hdf5_file_smaller_set = smaller_hdf5_file_dir / hdf5_file_whole_set.name

    # if the pdb file exists, copy the pdb file to smaller dataset.
    if hdf5_file_whole_set.exists():
        if not hdf5_file_smaller_set.exists():
            os.system(f"rsync -avz {hdf5_file_whole_set} {hdf5_file_smaller_set}")

        chain = pdb_full_id.split("_")[-1]
        f = h5py.File(hdf5_file_whole_set)

        # only in the required chain in the pdb file will count as the data point.
        if chain in f.keys():
            num_copied_data += 1
    return num_copied_data


def extract_dataset(dataset_csv: pd.DataFrame, hdf5_file_dir: Path,
                    smaller_hdf5_file_dir: Path, threshold: int,
                    dataset_name: str):
    """This function is used to extract (copy) hdf5 file from voxel_hdf5 to smaller_voxel_hdf5

    Args:
        dataset_csv: csv used as a reference to copy data.
        hdf5_file_dir: directory of voxel hdf5.
        smaller_hdf5_file_dir: directory of smaller voxel hdf5.
        threshold: threshold of number of data (in chain level).
        dataset_name: name of the aiming dataset.
    """
    logger.info(f"extracting {dataset_name} set.")
    tasks = []
    num_data = 0
    for idx, row in enumerate(dataset_csv.itertuples()):
        # if number of data (in chain level) exceeds the defined threshold, data set extraction complete.
        tasks.append(copy_data_instance.remote(hdf5_file_dir, smaller_hdf5_file_dir, row.full_id, row.id))
        if idx % 10 == 9:
            num_data_in_tasks = ray.get(tasks)
            print(len(tasks), num_data_in_tasks)
            num_data += np.sum(num_data_in_tasks)
            if num_data >= threshold:
                break


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

    # directory for hdf5 files
    hdf5_file_dir = root_dir / Path(args_data.hdf5_file_dir)

    # create a directory for new hdf5_files
    smaller_hdf5_file_dir = root_dir / Path(args_data.smaller_hdf5_file_dir)
    smaller_hdf5_file_dir.mkdir() if not smaller_hdf5_file_dir.exists() else None

    train_csv = dataset_csv.loc[dataset_csv["set"] == "training"]
    test_csv = dataset_csv.loc[dataset_csv["set"] == "test"]
    val_csv = dataset_csv.loc[dataset_csv["set"] == "validation"]

    # num_train_th, num_test_th, num_val_th = 80000, test_csv.size, 10000
    num_train_th, num_test_th, num_val_th = 80, 10, 10

    logger.info(f"Aimed training, test, val set data points: {num_train_th}, {num_test_th}, {num_val_th}")

    extract_dataset(dataset_csv=train_csv,
                    hdf5_file_dir=hdf5_file_dir,
                    smaller_hdf5_file_dir=smaller_hdf5_file_dir,
                    threshold=num_train_th,
                    dataset_name="train")

    extract_dataset(dataset_csv=test_csv,
                    hdf5_file_dir=hdf5_file_dir,
                    smaller_hdf5_file_dir=smaller_hdf5_file_dir,
                    threshold=num_test_th,
                    dataset_name="test")

    extract_dataset(dataset_csv=val_csv,
                    hdf5_file_dir=hdf5_file_dir,
                    smaller_hdf5_file_dir=smaller_hdf5_file_dir,
                    threshold=num_val_th,
                    dataset_name="validation")


if __name__ == "__main__":
    logger.info("Data gen started!")
    if not ray.is_initialized():
        ray.init(address='10.150.1.7:6379')
    data_gen_smaller()
