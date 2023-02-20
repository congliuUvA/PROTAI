"""This script is for box deduplication inside the dataset, also selecting the biological assembly files.

Please make sure to copying the hdf5 directory to hdd_store before running this script, for safety!"
"""
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
import numpy as np
import h5py
import ray
from utils import log

logger = log.get_logger(__name__)


def deduplicate(hdf5_file_path, pdb_id):
    # select the largest biological assembly file
    biological_assemblies_path = []
    biological_assemblies_size = []
    for pdb_hdf5 in hdf5_file_path.rglob(f"*{pdb_id}*"):
        biological_assemblies_path.append(pdb_hdf5)
        biological_assemblies_size.append(os.path.getsize(pdb_hdf5))
    selected_pdb_hdf5_file = biological_assemblies_path[np.argmax(biological_assemblies_size)]
    first_assembly_name = selected_pdb_hdf5_file.parent / f"{pdb_id}_pdb1.hdf5"

    # if there are multiple biological assembly files, select the largest one.
    if selected_pdb_hdf5_file != first_assembly_name:
        # swap the name of xxxx_pdbx.hdf5 with xxxx_pdb1.hdf5
        tmp_file_name = Path(selected_pdb_hdf5_file).parent / 'tmp.hdf5'
        os.system(f"mv {selected_pdb_hdf5_file} {tmp_file_name}")
        os.system(f"mv {first_assembly_name} {selected_pdb_hdf5_file}")
        os.system(f"mv {tmp_file_name} {first_assembly_name}")

    # de duplicate boxes in hdf5
    f = h5py.File(first_assembly_name, "a")
    box_hashmap = {}
    idx = 0
    for chain in f.keys():
        num_boxes = 0
        for box in f[chain]:
            data_string = "".join(f[chain][box][:].flatten().astype(str))
            if data_string not in box_hashmap:
                box_hashmap[data_string] = idx
                idx += 1
                num_boxes += 1
            else:
                f[chain].__delitem__(box)
        # record number of non-repeated boxes in one chain
        f[chain].attrs["num_boxes_chain"] = num_boxes
    # record number of non-repeated boxes in the struct
    f.attrs["num_boxes_struct"] = idx

    f.close()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(args: DictConfig):
    # load arguments of 3DCNN
    args_data = args.data

    args_data.hdf5_file_dir = "voxels_hdf5_test"

    parent_path = Path().cwd().parent
    hdf5_file_path = Path(parent_path) / args_data.hdf5_file_dir

    tasks = []
    # iterate through all the pdb1 hdf5 files
    for hdf5_file in hdf5_file_path.rglob("*.hdf5"):
        pdb_id = hdf5_file.name.split("_")[0]
        deduplicate(hdf5_file_path, pdb_id)


if __name__ == "__main__":
    logger.info("Deduplication started!")
    main()
