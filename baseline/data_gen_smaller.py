"""This module is for generating a smaller dataset 60000, 10000, 10000 for train/test/val dataset.

Only run this module after running deduplicate.py
"""

import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from utils import log
import h5py
import ray

logger = log.get_logger(__name__)


@ray.remote
def copy_data_instance(hdf5_file_dir, smaller_hdf5_file_dir, pdb_full_id, pdb_id):
    # always retrieve the largest file
    hdf5_file_whole_set = hdf5_file_dir / f"{pdb_id}_pdb1.hdf5"
    hdf5_file_smaller_set = smaller_hdf5_file_dir / hdf5_file_whole_set.name

    if not hdf5_file_whole_set.exists():
        return
    print(hdf5_file_whole_set)
    chain = pdb_full_id.split("_")[-1]
    try:
        f = h5py.File(hdf5_file_whole_set, "r")
    except OSError:
        return
    else:
        # only in the required chain in the pdb file will count as the data point.
        if chain in f.keys():
            # if the pdb file is not copied previously
            if not hdf5_file_smaller_set.exists():
                os.system(f"rsync {hdf5_file_whole_set} {hdf5_file_smaller_set}")
        f.close()

    return


def extract_dataset(dataset_csv: pd.DataFrame, hdf5_file_dir: Path,
                    smaller_hdf5_file_dir: Path, dataset_name: str):
    """This function is used to extract (copy) hdf5 file from voxel_hdf5 to smaller_voxel_hdf5

    Args:
        dataset_csv: csv used as a reference to copy data.
        hdf5_file_dir: directory of voxel hdf5.
        smaller_hdf5_file_dir: directory of smaller voxel hdf5.
        dataset_name: name of the aiming dataset.
    """
    logger.info(f"extracting {dataset_name} set.")
    tasks = []
    for idx, row in enumerate(dataset_csv.itertuples()):
        # if number of data (in chain level) exceeds the defined threshold, data set extraction complete.
        tasks.append(copy_data_instance.remote(hdf5_file_dir, smaller_hdf5_file_dir, row.full_id, row.id))
        # copy_data_instance(hdf5_file_dir, smaller_hdf5_file_dir, row.full_id, row.id)

    ray.get(tasks)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def data_gen_smaller(args: DictConfig):
    """This function generates hdf5 file for all pdb files"""
    # configuration assignment
    args_data = args.data
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
    train_csv = train_csv.loc[(train_csv["fold"] == "fold_0") |
                              (train_csv["fold"] == "fold_1") |
                              (train_csv["fold"] == "fold_2")]
    train_csv = train_csv.iloc[np.random.permutation(len(train_csv))]
    train_csv = train_csv.iloc[:100000]

    val_csv = dataset_csv.loc[dataset_csv["set"] == "validation"]

    # extract_dataset(dataset_csv=train_csv,
    #                 hdf5_file_dir=hdf5_file_dir,
    #                 smaller_hdf5_file_dir=smaller_hdf5_file_dir,
    #                 dataset_name="train")

    extract_dataset(dataset_csv=val_csv,
                    hdf5_file_dir=hdf5_file_dir,
                    smaller_hdf5_file_dir=smaller_hdf5_file_dir,
                    dataset_name="validation")

    # test_csv = dataset_csv.loc[dataset_csv["set"] == "test"]
    # extract_dataset(dataset_csv=test_csv,
    #                 hdf5_file_dir=hdf5_file_dir,
    #                 smaller_hdf5_file_dir=smaller_hdf5_file_dir,
    #                 dataset_name="test")


if __name__ == "__main__":
    logger.info("Data gen started!")
    if not ray.is_initialized():
        ray.init(address='10.150.1.8:6379')
    np.random.seed(0)
    data_gen_smaller()
