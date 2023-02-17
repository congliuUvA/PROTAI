"""This module is for voxel box generation."""
import Bio
import numpy as np
from numpy import ndarray
import ray
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from Bio.PDB.NeighborSearch import NeighborSearch
from itertools import product
from Bio.PDB.vectors import Vector, rotmat, rotaxis2m
from Bio.PDB.Atom import Atom
import freesasa
import h5py
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
import os

num_of_voxels = 20
len_of_voxel = 0.8

RES_NAME = ["ALA", "ARG", "ASN", "ASP", "CYS",
            "GLU", "GLN", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL"]


def pqr2pdb(fname_pqr, fname_pdb):

    lines = open(fname_pqr, 'r').readlines()
    fout = open(fname_pdb, 'w')

    chain_id = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    ich = 0
    for line in lines:
        content = line.split()
        if line[:6] in ['ATOM  ', 'HETATM']:
            atnm = line[13:15].upper()
            resName = line[17:20].upper()
            symbol = atnm[0]
            if atnm[:2] in ['CL', 'NA', 'MG', 'BE', 'LI', 'ZN']:
                symbol = atnm[:2]
            elif atnm[:2] == 'CA' and resName[:2] == 'CA':
                symbol = 'CA'
            sline = line[:21] + chain_id[ich] + line[22:54] + \
                "  " + content[-2] + "  " + content[-1] + "          %2s  " % symbol
            print(sline, file=fout)
        elif line[:3] == 'END':
            print('END', file=fout)
        else:
            print(line[:-1], file=fout)

        if line[:3] == 'TER':
            ich += 1

    fout.close()


def load_protein(arguments, pdb_name: str, file_path: str) -> Bio.PDB.Structure.Structure:
    """This function is used to load the protein file in format of mmCIF to the structure in Biopython.

    Args:
        arguments: arguments input by users.
        pdb_name: PDB ID
        file_path: file path of the protein file.

    Returns:
        struct: Structure of the selected protein in Biopython.
    """
    if arguments.addH:
        pqr_file_path = str(Path(file_path).parent.joinpath(pdb_name)) + ".pqr"
        os.system(f"pdb2pqr30 --ff=PARSE {file_path} {pqr_file_path}")
        pqr2pdb(pqr_file_path, Path(file_path).stem + "_pqr.pdb")
        file_path = Path(file_path).stem + "_pqr.pdb"

    parser = PDBParser(QUIET=1) if "pdb" in Path(file_path).suffix else MMCIFParser(QUIET=1)
    struct = parser.get_structure(pdb_name, file_path)
    return struct


def range_editor(coord: ndarray) -> List:
    """This function edits the range of the coordinates,
    given the central position of the ca atom under the cartesian coordinates.
    Args:
        coord: central ca coords.
    Returns:
        range_list: list of x,y,z range
    """
    x, y, z = coord[0], coord[1], coord[2]
    range_x = [x - num_of_voxels / 2 * len_of_voxel, x + num_of_voxels / 2 * len_of_voxel]
    range_y = [y - num_of_voxels / 2 * len_of_voxel, y + num_of_voxels / 2 * len_of_voxel]
    range_z = [z - num_of_voxels / 2 * len_of_voxel, z + num_of_voxels / 2 * len_of_voxel]
    range_list = [range_x, range_y, range_z]
    return range_list


def cal_projected_cb_coords(c_atom, n_atom, ca_atom) -> ndarray:
    """
    This function calculate the projected CB coordinates by given C, N, CA in the same residue.
    Procedure can be find in word file.
    Args:
        c_atom: carbon atom in biopython.
        n_atom: nitrogen atom.
        ca_atom: carbon alpha atom.
    Returns:
        cb_atom_coord: projected CB atom coordinates
    """
    p_vector = c_atom.coord - n_atom.coord
    q_vector = n_atom.coord - ca_atom.coord
    u_vector = (p_vector + q_vector) / np.linalg.norm(p_vector + q_vector)
    r_vector = (p_vector - q_vector) / np.linalg.norm(p_vector - q_vector)
    m = rotaxis2m(-np.pi * 125 / 360, Vector(r_vector))
    b_vector = Vector(u_vector).left_multiply(m) / np.linalg.norm(Vector(u_vector).left_multiply(m))
    cb_atom_coord = b_vector.get_array() * 1.5 + ca_atom.coord
    return cb_atom_coord


def gen_ca_cb_vectors(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List, List]:
    """ This function generates ca_list, cb_list and ca_cb vectors for all residues in one pbd file.
    Args:
        struct: structure in biopython.

    Returns:

    """
    # examine whether all central atom come from the normal amino acids
    ca_cb_vectors = []
    ca_list, cb_list = [], []
    for res in struct.get_residues():
        # skip residue that are not AA.
        if res.get_resname() in RES_NAME:
            ca_atom, c_atom, n_atom, real_cb_atom, projected_cb_atom, cb_atom_coord = [None] * 6
            for atom in res.get_atoms():
                if atom.get_name() == "CA": ca_atom = atom
                if atom.get_name() == "C": c_atom = atom
                if atom.get_name() == "N": n_atom = atom
                if atom.get_name() == "CB": real_cb_atom = atom

            if (ca_atom is not None) and (c_atom is not None) and (n_atom is not None):
                if real_cb_atom is not None:
                    # calculate projected CB coordinates
                    cb_atom_coord = cal_projected_cb_coords(c_atom, n_atom, ca_atom)
                    projected_cb_atom = Atom(name="CB_fake", coord=cb_atom_coord, bfactor=real_cb_atom.bfactor,
                                             occupancy=real_cb_atom.occupancy, altloc=real_cb_atom.altloc,
                                             fullname=real_cb_atom.fullname,
                                             serial_number=real_cb_atom.serial_number, element="C", )
                    projected_cb_atom.set_parent(res)
                else:
                    # calculate projected CB coordinates
                    cb_atom_coord = cal_projected_cb_coords(c_atom, n_atom, ca_atom)
                    # create cb atom
                    projected_cb_atom = Atom(name="CB_fake", coord=cb_atom_coord, bfactor=0, occupancy=0,
                                             altloc="", fullname="", serial_number="", element="C", )
                    projected_cb_atom.set_parent(res)

                ca_list.append(ca_atom)
                cb_list.append(projected_cb_atom)
                ca_cb_vectors.append(Vector(ca_atom.coord - cb_atom_coord))
    return ca_list, cb_list, ca_cb_vectors


def select_in_range_atoms(
        struct, ca_list, cb_list, ca_cb_vectors,
        selected_central_atom_type="CA", shift=0
) -> Tuple[List, List, List]:
    """This function generates the selected in-range atom corresponding to the central atom ["CA", "CB"].
    Args:
        struct: structure in biopython.
        ca_list: list containing CA in the PDB file.
        cb_list: list containing CB in the PDB file.
        ca_cb_vectors: list containing ca_cb vectors.
        selected_central_atom_type: string stating which central atom is selected.
        shift: shift of carbon alpha along the diagonal direction.
    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        rot_mat: list of rotation matrix corresponding to every residue.
        central_atom_coords: list of central atom coordinate after applying shift.
    """
    central_atom_list = ca_list if selected_central_atom_type == "CA" else cb_list

    # get corner vector
    corner_vector = Vector(1, 1, 1)

    voxel_atom_lists = []

    # search atoms that are within the range
    nb_search = NeighborSearch(list(struct.get_atoms()))

    # record rotation matrix for every residue.
    rot_mats = []

    # record central atom coordinates
    central_atom_coords = []

    # searching for atoms that are within the range
    for central_atom, ca_cb_vector in zip(central_atom_list, ca_cb_vectors):
        # the first atom in the voxel is the central atom.
        voxel_atom_list = [central_atom]

        # generate rotation matrix
        # same rotation for all the atoms from one residue
        rot = np.identity(3) if selected_central_atom_type == "CA" else rotmat(ca_cb_vector, corner_vector)

        # Shift of Ca position can be applied by confirming the shift of the central atom, i.e. by confirming the new
        # central atom coordinates, vector subtraction
        central_atom_coord = central_atom.get_coord() - shift * ca_cb_vector.normalized().get_array()

        # search in-sphere atoms
        # It doesn't matter if not transferring atom coordinate here because sphere is invariant.
        searched_atom_list = nb_search.search(central_atom_coord, np.sqrt(3 * 10 ** 2) * 0.8)

        # keep in-range atoms
        for atom in searched_atom_list:
            # remember to left-multiply rotation matrix all the time when requiring atom coordinates.
            central_range_list = range_editor(central_atom_coord)
            atom_coord = rot @ (atom.get_coord() - central_atom_coord) + central_atom_coord
            range_x, range_y, range_z = central_range_list[0], central_range_list[1], central_range_list[2]
            atom_x, atom_y, atom_z = atom_coord[0], atom_coord[1], atom_coord[2]
            if range_x[0] < atom_x < range_x[1] and \
                    range_y[0] < atom_y < range_y[1] and \
                    range_z[0] < atom_z < range_z[1]:
                # Do not contain central residue in the box.
                if atom != central_atom and atom.parent != central_atom.parent:
                    voxel_atom_list.append(atom)

            # if atom.parent == central_atom.parent and not (
            #         range_x[0] < atom_x < range_x[1] and
            #         range_y[0] < atom_y < range_y[1] and
            #         range_z[0] < atom_z < range_z[1]
            # ):
            #     raise ValueError("Shift value needs to be justified,"
            #                      "atoms from central residual is not included in the box! "
            #                      "Please use a different shift value.")

        voxel_atom_lists.append(voxel_atom_list)

        # save rotation matrix
        rot_mats.append(rot)

        # save central atom coordinate
        central_atom_coords.append(central_atom_coord)

    return voxel_atom_lists, rot_mats, central_atom_coords


def generate_voxel_atom_lists(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List, List]:
    """This function generates permitted range for every selected CA. (20 * 20 * 20), 0.8 A for dimension.
    Args:
        struct: structure in biopython.
    """
    ca_list, cb_list, ca_cb_vectors = gen_ca_cb_vectors(struct)
    voxel_atom_lists, rot_mats, central_atom_coords = select_in_range_atoms(
        struct, ca_list, cb_list, ca_cb_vectors, selected_central_atom_type="CB", shift=0
    )
    return voxel_atom_lists, rot_mats, central_atom_coords


def generate_selected_element_voxel(
        arguments, elements_list: List, selected_element: str, voxel_atom_list: List,
        rot_mat: np.array, central_atom_coord: np.array, sasa_results: freesasa.Result,
) -> Tuple[ndarray, ndarray, ndarray]:
    """This function generate selected-element voxel by inserting True/False to 20*20*20 voxel.
    Args:
        arguments: arguments of the user.
        elements_list: elments in interest
        selected_element: selected element for one single channel, "C", "N", "O", "S".
        voxel_atom_list: The pre-generated voxel list containing central CA atoms and the surrounding atoms.
        rot_mat: rotation matrix for the selected residue.
        central_atom_coord: list of central atom coordinate after applying shift.
        sasa_results: results of solvent accessible surface area.
    """
    # 1. create True False voxel.
    voxels_bool = (np.ones((num_of_voxels, num_of_voxels, num_of_voxels)) == 0).astype(bool)
    # create partial_charges array
    partial_charges = np.zeros((num_of_voxels, num_of_voxels, num_of_voxels))
    # create solvent accessible surface area array
    sasa = np.zeros((num_of_voxels, num_of_voxels, num_of_voxels))

    # 2. selected corresponding atoms
    selected_atom_list = []
    # the first atom in voxel_atom_list is the central atom
    for atom in voxel_atom_list[1:]:
        if atom.element == selected_element and atom.element in elements_list:
            selected_atom_list.append(atom)

    # 3. Generate the coordinates of 20 * 20 * 20 voxels,
    # 1 centric coordinate for 1 voxel, (20, 20, 20, 3)
    initial_atom_coord = central_atom_coord - num_of_voxels / 2 * len_of_voxel + len_of_voxel / 2
    voxels_coords = [[[[
        initial_atom_coord[0] + i * len_of_voxel,
        initial_atom_coord[1] + j * len_of_voxel,
        initial_atom_coord[2] + k * len_of_voxel]
        for i in range(num_of_voxels)]
        for j in range(num_of_voxels)]
        for k in range(num_of_voxels)]
    voxels_coords = np.array(voxels_coords)

    # 4. Add atom to the voxel.
    for atom in selected_atom_list:
        atom_coord = rot_mat @ (atom.coord - central_atom_coord) + central_atom_coord
        voxels_bool, partial_charges, sasa = add_atom_to_voxel(
            arguments, atom, atom_coord, voxels_coords, voxels_bool, partial_charges, sasa_results, sasa
        )
    return voxels_bool, partial_charges, sasa


def gen_one_voxel_coords(coord: np.array) -> np.array:
    """This function generates the eight coordinates
    of one single voxel given the coordinates of left corner of the voxel.
    Args:
        coord: coordinate of the most left and inner vertex of the voxel.
    Returns:
        voxel_coords: voxel coordinates of eight vertices.
    """
    # 8 coordinates
    factor_list = np.array(list(product("01", repeat=3))).astype(np.int32)
    voxel_coords = factor_list * len_of_voxel + coord.T
    return voxel_coords


def add_atom_to_voxel(
        arguments,
        atom: Bio.PDB.Atom.Atom,
        atom_coord: np.array,
        voxels_coords: ndarray,
        voxels_bool: ndarray,
        partial_charges: ndarray,
        sasa_results: freesasa.Result,
        sasa: ndarray,
) -> Tuple[ndarray, ndarray, ndarray]:
    """This function adds atom to the corresponding location of the voxel.
    Args:
        arguments: arguments of the users.
        atom: atom object to be added
        atom_coord: atom coordinate of atom to be added
        voxels_coords: coordinates of the 20*20*20 voxels.
        voxels_bool: voxel array in bool.
        partial_charges: partial_charge channel, 20*20*20 array.
        sasa_results: results of solvent accessible surface area.
        sasa: solvent accessible surface area array.

    Returns:
        voxels_bool: voxel filled in new bool values.
        partial_charges: filled partial charge channel, 20*20*20 array
        sasa: solvent accessible surface area array.
    """

    radius_table = {"C": 1.7, "O": 1.37, "N": 1.45, "P": 1.49, "S": 1.7, "H": 1.0}

    dists_element_voxels_coords = np.sqrt(np.sum((voxels_coords - atom_coord) ** 2, axis=-1))
    # (20, 20, 20)
    contact_element_voxel_coords = np.where(dists_element_voxels_coords < radius_table[atom.element])
    add_bool = contact_element_voxel_coords
    voxels_bool[add_bool[0].T, add_bool[1].T, add_bool[2].T] = True
    if arguments.add_partial_charges:
        partial_charges[add_bool[0].T, add_bool[1].T, add_bool[2].T] = atom.occupancy
    if arguments.add_sasa:
        sasa[add_bool[0].T, add_bool[1].T, add_bool[2].T] = sasa_results.atomArea(atom.get_serial_number() - 1)

    return voxels_bool, partial_charges, sasa


def visualize_voxels(arguments, voxel_list: List):
    """This function visualizes the voxels of CNOS.
    Args:
        arguments: arguments input by user
        voxel_list: a list include all voxels of 4 channels.
    """
    voxelarray = voxel_list[0] | voxel_list[1] | voxel_list[2] | voxel_list[3]
    voxelarray = voxelarray | voxel_list[4] if arguments.addH else voxelarray
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxel_list[0]] = "green"
    colors[voxel_list[1]] = "blue"
    colors[voxel_list[2]] = "red"
    colors[voxel_list[3]] = "yellow"
    if arguments.addH:
        colors[voxel_list[4]] = "purple"
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()


def cal_sasa(struct, pdb_name):
    """Calculate sasa results for the given structure."""
    # calculate sasa
    all_radiis = []
    all_coords = [atom.coord for atom in struct.get_atoms()]

    # requiring b_factor (b_factor cannot be obtained by atom.bfactor due to the biopython reason)
    pqr2pdb_file_path = pdb_name + "_pqr.pdb"
    with open(pqr2pdb_file_path, "r") as f:
        for line in f.readlines():
            content = line.split()
            if len(content) > 1:
                all_radiis.append(float(content[-2]))

    sasa_results = freesasa.calcCoord(np.array(all_coords).flatten(), all_radiis)
    return sasa_results


def gen_voxel_binary_array(arguments, f, struct, pdb_name,
                           voxel_atom_lists: List, rot_mats: List,
                           central_atom_coords: List):
    """This function generates voxel binary array given the
    selected voxel atom lists and rotation matrix and central atom coordinates

    Args:
        arguments: arguments of the user.
        f: hdf5 file for the pdb
        struct: PDB structure in biopython.
        pdb_name: PDB ID of interest.
        voxel_atom_lists: voxel atom lists of each voxel box. (len = num of residues in total)
        rot_mats: rotation matrix of each voxel box. (len = num of residues in total)
        central_atom_coords: central atom coordinates of each box.
    """
    for idx, voxel_atom_list, rot_mat, central_atom_coord in tqdm(zip(
            np.arange(len(voxel_atom_lists)), voxel_atom_lists, rot_mats, central_atom_coords
    ), total=len(voxel_atom_lists)):
        # take out the central atom
        central_atom = voxel_atom_list[0]

        # iterate through ["C", "N", "O", "S", "H"]
        elements_list = ["C", "N", "O", "S", "H"] if arguments.addH else ["C", "N", "O", "S"]

        # calculate sasa
        sasa_results = cal_sasa(struct, pdb_name) if arguments.add_sasa else None

        all_voxel = []
        all_partial_charges = []
        all_sasa = []
        for element in elements_list:
            selected_element_voxel, partial_charges, sasa, = generate_selected_element_voxel(
                arguments,
                elements_list,
                element,
                voxel_atom_list,
                rot_mat,
                central_atom_coord,
                sasa_results,
            )
            all_voxel.append(selected_element_voxel)
            all_partial_charges.append(partial_charges)
            all_sasa.append(sasa)

        # visualization
        # visualize_voxels(arguments, all_voxel)

        # store voxel, partial_charges and sasa as file format of hdf5
        # binary voxel box for 4 channels
        voxel_per_residue = np.array(all_voxel, dtype=np.bool_)  # (4, 20, 20, 20)

        # metadata
        pdb_id = central_atom.parent.get_full_id()[0]
        chain_id = central_atom.parent.parent.id
        residue_name = central_atom.parent.get_resname()
        residue_serial_number = central_atom.parent.get_full_id()[-1][-2]
        residue_position = central_atom.coord
        residue_icode = central_atom.parent.get_full_id()[-1][-1]

        # chain_group creation
        group = f.create_group(chain_id) if chain_id not in f else f[chain_id]
        dataset = group.create_dataset(str(idx), data=voxel_per_residue, compression="lzf")

        # attributes generation
        dataset.attrs["pdb_id"] = pdb_id
        dataset.attrs["chain_id"] = chain_id
        dataset.attrs["residue_name"] = residue_name
        dataset.attrs["residue_serial_number"] = residue_serial_number
        dataset.attrs["residue_position"] = residue_position
        dataset.attrs["residue_icode"] = residue_icode


def count_res(struct: Bio.PDB.Structure.Structure) -> int:
    """Count number of residues in the given structure."""
    num = 0
    for res in struct.get_residues():
        if res.get_resname() in RES_NAME:
            ca_atom, c_atom, n_atom = [None] * 3
            for atom in res.get_atoms():
                if atom.get_name() == "C": c_atom = atom
                if atom.get_name() == "N": n_atom = atom
                if atom.get_name() == "CA": ca_atom = atom
            if (ca_atom is not None) and (c_atom is not None) and (n_atom is not None):
                num += 1
    return num


@ray.remote
def gen_voxel_box_file(arguments, idx):
    """The main function of generating voxels.

    Args:
        arguments: arguments input from user.
    """
    # configuration set up
    pdb_name = arguments.pdb_name
    pdb_path = str(Path.cwd().joinpath(arguments.pdb_path))
    pdb_id = arguments.pdb_id

    # Load protein structure
    struct = load_protein(arguments, pdb_name, pdb_path)

    # count number if residues in the structure.
    num_residues = count_res(struct)

    # start a hdf5 file
    f = h5py.File(str(Path(arguments.hdf5_file_dir) / pdb_id) + ".hdf5", "a")

    print(f"Dealing with file index: {idx}, {str(Path(arguments.hdf5_file_dir) / pdb_id) + '.hdf5'}")

    # count number of dataset if hdf5 file
    num_datasets = len([box for chain in f.keys() for box in f[chain]])
    f.close()
    # if the hdf5 file is completed, skip the function
    if num_datasets == num_residues:
        print(f"skip {str(Path(arguments.hdf5_file_dir) / pdb_id)}")
        return

    else:
        # f = h5py.File(str(Path(arguments.hdf5_file_dir) / pdb_id) + ".hdf5", "w")
        # # generate atom lists for 20*20*20 voxels, num_of_residue in pdb file in total.
        # (
        #     voxel_atom_lists, rot_mats, central_atom_coords
        # ) = generate_voxel_atom_lists(struct) # (num_ca, num_atoms_in_voxel)
        #
        # gen_voxel_binary_array(arguments, f, struct, pdb_name,
        #                        voxel_atom_lists, rot_mats, central_atom_coords)
        # f.close()
        print(f"not the same!")
        return
