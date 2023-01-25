"""This script is used for hdf5 data generation for all pdb files in Protein Data Bank."""
import hydra
from omegaconf import DictConfig
from pathlib import Path
import gzip
import shutil
from voxel_box.voxel_rotate_atom import gen_voxel_box_file


@hydra.main(version_base=None, config_path="../config", config_name="config")
def data_gen(args: DictConfig):
    """This function generates hdf5 file for all pdb files"""
    # configuration assignment
    args_data = args.data
    args_voxel_box = args.voxel_box
    parent_dir = Path.cwd().parent

    # raw pdb file path
    raw_pdb_dir = parent_dir / Path(args_data.raw_pdb_dir)

    # create a directory for hdf5 filess
    hdf5_file_dir = parent_dir / Path("voxels_hdf5")
    hdf5_file_dir.mkdir() if not hdf5_file_dir.exists() else None

    # assign path to arguments of voxel_box
    args_voxel_box.hdf5_file_dir = str(hdf5_file_dir)

    for pdb in raw_pdb_dir.rglob("*.gz"):
        # unzip gz file
        with gzip.open(pdb, 'rb') as f_in:
            pdb_unzip = ".".join(str(pdb).split(".")[:-1])
            with open(pdb_unzip, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # assign pdb info to args_voxel_box
        args_voxel_box.pdb_name = Path(pdb_unzip).stem
        args_voxel_box.pdb_path = pdb_unzip
        gen_voxel_box_file()
        break


if __name__ == "__main__":
    data_gen()
