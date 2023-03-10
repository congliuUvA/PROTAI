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
    root_dir = baseline_dir.parent if not args_data.use_hddstore else "/ssdstore/cliu3"

    # raw pdb file path
    raw_pdb_dir = baseline_dir.parent / Path(args_data.raw_pdb_dir)

    # dataset split csv path
    dataset_split_csv = baseline_dir.parent / Path(args_data.dataset_split_csv)

    # download raw data set if not existed in the WR.
    if not raw_pdb_dir.exists():
        raw_pdb_dir.mkdir()
        logger.info("Start downloading raw pdb files...")
        os.system("sh ../../download/rsyncPDB.sh")
        logger.info('Finished downloading!')

    # load gz file list
    gz_npy_path = baseline_dir / "gz_file_list.npy"
    gz_file_list = np.load(str(gz_npy_path), allow_pickle=True) if gz_npy_path.exists() \
        else np.array([pdb for pdb in raw_pdb_dir.rglob("*.gz")])
    np.save(str(gz_npy_path), gz_file_list) if not gz_npy_path.exists() else None

    # decide how many files in gz_file_list to generate
    gz_file_list = gz_file_list[:int(len(gz_file_list) * args_data.proportion_pdb)]
    np.random.seed(0)
    np.random.shuffle(gz_file_list)

    logger.info(f"Processing {len(gz_file_list)} pdb files!")

    # create a directory for hdf5 files
    hdf5_file_dir = root_dir / Path(args_data.hdf5_file_dir)
    hdf5_file_dir.mkdir() if not hdf5_file_dir.exists() else None

    # assign path to arguments of voxel_box
    args_voxel_box.hdf5_file_dir = str(hdf5_file_dir)

    dataset_split_pd = pd.read_csv(str(dataset_split_csv))
    pdb_id_array = np.unique(np.array(dataset_split_pd.id))

    idx, total = 0, gz_file_list.shape[0]

    start_end_idx = [int(i * total / args_data.num_partition) for i in range(args_data.num_partition + 1)]
    if not hasattr(args_data, "partition_idx"):
        logger.debug("Please specify the partition index in command line!")
    start, end = start_end_idx[args_data.partition_idx - 1], start_end_idx[args_data.partition_idx]

    # ray tasks
    logger.info("Start ray tasks.")
    tasks = []
    logger.info(gz_file_list[start: end].shape)

    for pdb in gz_file_list[start: end]:
        # unzipped pdb file name
        pdb_pure_id = pdb.name.split(".")[0]
        assembly_id = pdb.name.split(".")[1]
        pdb_id = pdb_pure_id + "_" + assembly_id  # e.g. "2HBS_pdb1"

        # if pdb id is not in the list, skip the pdb file.
        if pdb_pure_id in pdb_id_array:
            logger.info(f"Dealing with file index: {idx}, {str(Path(hdf5_file_dir) / pdb_id) + '.hdf5'}")
            # unzip pdb file
            pdb_unzip = ".".join(str(pdb).split(".")[:-1])
            os.system(f"gunzip -c {pdb} > {pdb_unzip}")

            # assign pdb info to args_voxel_box
            args_voxel_box.pdb_name = pdb_pure_id
            args_voxel_box.pdb_path = pdb_unzip
            args_voxel_box.pdb_id = pdb_id

            task = gen_voxel_box_file.remote(args_voxel_box)
            # gen_voxel_box_file(args_voxel_box)

            tasks.append(task)

            idx += 1

    ray.get(tasks)
    logger.info(f"{args_data.partition_idx} / {args_data.num_partition} completed!")


if __name__ == "__main__":
    logger.info("Data gen started!")
    if not ray.is_initialized():
        ray.init(address='auto')
    data_gen()
