"""This script is used for hdf5 data generation for all pdb files in Protein Data Bank."""
import hydra
from omegaconf import DictConfig
from pathlib import Path
from voxel_box.voxel_rotate_atom import gen_voxel_box_file
import os
import pandas as pd
import numpy as np
import ray
from utils import log

logger = log.get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def data_gen(args: DictConfig):
    """This function generates hdf5 file for all pdb files"""
    # configuration assignment
    args_data = args.data
    args_voxel_box = args.voxel_box
    baseline_dir = Path.cwd()  # baseline/
    root_dir = baseline_dir.parent if not args_data.use_hddtore else "/hddstore/cliu3"

    # raw pdb file path
    raw_pdb_dir = root_dir / Path(args_data.raw_pdb_dir)

    # dataset split csv path
    dataset_split_csv = root_dir / Path(args_data.dataset_split_csv)

    # download raw data set if not existed in the WR.
    if not raw_pdb_dir.exists():
        raw_pdb_dir.mkdir()
        logger.info("Start downloading raw pdb files...")
        os.system("sh ../../download/rsyncPDB.sh")
        logger.info('Finished downloading!')

    # create a directory for hdf5 files
    hdf5_file_dir = root_dir / Path(args_data.hdf5_file_dir)
    hdf5_file_dir.mkdir() if not hdf5_file_dir.exists() else None

    # assign path to arguments of voxel_box
    args_voxel_box.hdf5_file_dir = str(hdf5_file_dir)

    dataset_split_pd = pd.read_csv(str(dataset_split_csv))
    pdb_id_array = np.unique(np.array(dataset_split_pd.id))

    idx = 0
    total = 289756

    # ray tasks
    logger.info("Start ray tasks.")
    tasks = []
    for pdb in raw_pdb_dir.rglob("*.gz"):
        #  1/2
        # if idx > int(total / 2):
        #     print("1/2 completed")
        #     break

        # # 2/2
        # if idx <= int(total / 2):
        #     idx += 1
        #     continue

        # unzipped pdb file name
        pdb_pure_id = pdb.name.split(".")[0]
        assembly_id = pdb.name.split(".")[1]
        pdb_id = pdb_pure_id + "_" + assembly_id  # e.g. "2HBS_pdb1"

        # if pdb id is not in the list, skip the pdb file.
        if pdb_pure_id not in pdb_id_array:
            continue

        # unzip pdb file
        pdb_unzip = ".".join(str(pdb).split(".")[:-1])
        os.system(f"gunzip -c {pdb} > {pdb_unzip}")

        # assign pdb info to args_voxel_box
        args_voxel_box.pdb_name = pdb_pure_id
        args_voxel_box.pdb_path = pdb_unzip
        args_voxel_box.pdb_id = pdb_id
        # task = gen_voxel_box_file.remote(args_voxel_box, idx)
        gen_voxel_box_file(args_voxel_box, idx)
        # remove generated pdb file, clean up the mess
        # os.system(f"rm {pdb_unzip}")
        # tasks.append(task)
        idx += 1

    ray.get(tasks)


if __name__ == "__main__":
    logger.info("Data gen started!")
    if not ray.is_initialized():
        ray.init(address='10.150.1.7:6379')
    data_gen()
