"""This script is used for hdf5 data generation for all pdb files in Protein Data Bank."""
import hydra
from omegaconf import DictConfig
from pathlib import Path
from voxel_rotate_atom import gen_voxel_box_file
import os


@hydra.main(version_base=None, config_path="../config", config_name="config")
def data_gen(args: DictConfig):
    """This function generates hdf5 file for all pdb files"""
    # configuration assignment
    args_data = args.data
    args_voxel_box = args.voxel_box
    baseline_dir = Path.cwd().parent  # baseline/
    root_dir = baseline_dir.parent
    # raw pdb file path
    raw_pdb_dir = root_dir / Path(args_data.raw_pdb_dir)

    # create a directory for hdf5 filess
    hdf5_file_dir = baseline_dir / Path("voxels_hdf5")
    hdf5_file_dir.mkdir() if not hdf5_file_dir.exists() else None

    # assign path to arguments of voxel_box
    args_voxel_box.hdf5_file_dir = str(hdf5_file_dir)

    for pdb in raw_pdb_dir.rglob("*.gz"):
        # unzipped pdb file name
        pdb_id = pdb.name.split(".")[0]
        if (hdf5_file_dir.joinpath(pdb_id + ".hdf5")).exists():
            continue
        pdb_unzip = ".".join(str(pdb).split(".")[:-1])
        os.system("gunzip --keep -f " + str(pdb))

        # assign pdb info to args_voxel_box
        args_voxel_box.pdb_name = Path(pdb_unzip).stem
        args_voxel_box.pdb_path = pdb_unzip
        gen_voxel_box_file(args_voxel_box)


if __name__ == "__main__":
    data_gen()
