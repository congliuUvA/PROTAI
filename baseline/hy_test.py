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
