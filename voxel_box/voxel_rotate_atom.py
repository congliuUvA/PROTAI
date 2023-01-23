"""This module is for voxel box generation."""

from examination import load_protein
import Bio
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from Bio.PDB.NeighborSearch import NeighborSearch
from itertools import product
from Bio.PDB.vectors import Vector, rotmat

num_of_voxels = 20
len_of_voxel = 0.8


def range_editor(coord: ndarray) -> List:
    """This function edits the range of the coordinates,
    given the central position of the ca atom under the cartesian coordinates.
    Args:
        coord: central ca coords.
    Returns:
        range_list: list of x,y,z range
    """
    x, y, z = coord[0], coord[1], coord[2]
    range_x = [x - num_of_voxels/2 * len_of_voxel, x + num_of_voxels/2 * len_of_voxel]
    range_y = [y - num_of_voxels/2 * len_of_voxel, y + num_of_voxels/2 * len_of_voxel]
    range_z = [z - num_of_voxels/2 * len_of_voxel, z + num_of_voxels/2 * len_of_voxel]
    range_list = [range_x, range_y, range_z]
    return range_list


def generate_central_atoms(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List]:
    """This function generates central atom lists and their corresponding ranges.
    Args:
        struct: structure in biopython.
    Returns:
        ca_list
        cb_list
    """
    ca_list, cb_list = [], []
    for res in struct.get_residues():
        for atom in res.get_atoms():
            if atom.get_parent().get_resname() == "GLY":
                if atom.get_name() == "CA":
                    ca_list.append(atom)
                    cb_list.append(atom)
                    continue

            if atom.get_name() == "CA":
                ca_list.append(atom)
            if atom.get_name() == "CB":
                cb_list.append(atom)

    # sampling
    np.random.seed(0)
    ca_sample, cb_sample = [], []
    num_res = len(ca_list)
    sample_num = min(num_res // 2, 100)
    sample_index = np.random.randint(0, num_res-1, sample_num)
    for index in sample_index:
        ca_sample.append(ca_list[index])
        cb_sample.append(cb_list[index])

    return ca_sample, cb_sample


def select_in_range_atoms(
        struct, ca_list, cb_list,
        selected_central_atom_type="CA", shift=0
) -> Tuple[List, List, List]:
    """This function generates the selected in-range atom corresponding to the central atom ["CA", "CB"].
    Args:
        struct: structure in biopython.
        ca_list: list containing CA in the PDB file.
        cb_list: list containing CB in the PDB file.
        selected_central_atom_type: string stating which central atom is selected.
        shift: shift of carbon alpha along the diagonal direction.
    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        rot_mat: list of rotation matrix corresponding to every residue.
        central_atom_coords: list of central atom coordinate after applying shift.
    """
    central_atom_list = ca_list if selected_central_atom_type == "CA" else cb_list

    # get ca_cb vector and corner vector
    ca_cb_vectors = [Vector(ca.get_coord()) - Vector(cb.get_coord()) for ca, cb in zip(ca_list, cb_list)]
    corner_vector = Vector(1, 1, 1)

    voxel_atom_lists = []

    # search atoms that are within the range
    nb_search = NeighborSearch(list(struct.get_atoms()))

    # record rotation matrix for every residue.
    rot_mats = []

    # record central atom coordinates
    central_atom_coords = []

    # searching for atoms that are within the range
    for central_atom, ca_cb_vector in tqdm(zip(central_atom_list, ca_cb_vectors), total=len(ca_cb_vectors)):
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
                if atom != central_atom and atom.parent != central_atom.parent:
                    voxel_atom_list.append(atom)

            if atom.parent == central_atom.parent and not (
                    range_x[0] < atom_x < range_x[1] and
                    range_y[0] < atom_y < range_y[1] and
                    range_z[0] < atom_z < range_z[1]
            ):
                raise ValueError("Shift value needs to be justified,"
                                 "atoms from central residual is not included in the box! "
                                 "Please use a different shift value.")

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
    ca_list, cb_list = generate_central_atoms(struct)
    voxel_atom_lists, rot_mats, central_atom_coords = select_in_range_atoms(
        struct, ca_list, cb_list, selected_central_atom_type="CB", shift=0
    )
    return voxel_atom_lists, rot_mats, central_atom_coords


def generate_selected_element_voxel(
        selected_element: str, voxel_atom_list: List, rot_mat: np.array, central_atom_coord: np.array
):
    """This function generate selected-element voxel by inserting True/False to 20*20*20 voxel.
    Args:
        selected_element: selected element for one single channel, "C", "N", "O", "S".
        voxel_atom_list: The pre-generated voxel list containing central CA atoms and the surrounding atoms.
        rot_mat: rotation matrix for the selected residue.
        central_atom_coord: list of central atom coordinate after applying shift.
    """
    if selected_element not in ["C", "N", "O", "S", "H"]:
        raise ValueError("'selected_element' has to be in the options of 'C', 'N', 'S', 'O', 'H'")

    # 1. create True False voxel.
    voxels_bool = (np.ones((num_of_voxels, num_of_voxels, num_of_voxels)) == 0).astype(bool)

    # 2. selected corresponding atoms
    selected_atom_list = []
    for atom in voxel_atom_list:
        if atom.element == selected_element and atom.element in ["C", "N", "O", "S", "H"]:
            selected_atom_list.append(atom)

    # 3. Generate the coordinates of 20 * 20 * 20 voxels,
    # 1 centric coordinate for 1 voxel, (20, 20, 20, 3)
    initial_atom_coord = central_atom_coord - num_of_voxels/2 * len_of_voxel + len_of_voxel/2
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
        voxels_bool = add_atom_to_voxel(atom, atom_coord, voxels_coords, voxels_bool)
    return voxels_bool


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
        atom: Bio.PDB.Atom.Atom,
        atom_coord: np.array,
        voxels_coords: ndarray,
        voxels_bool: ndarray,
) -> ndarray:
    """This function adds atom to the corresponding location of the voxel.
    Args:
        atom: atom object to be added
        atom_coord: atom coordinate of atom to be added
        voxels_coords: coordinates of the 20*20*20 voxels.
        voxels_bool: voxel array in bool.
    Returns:
        voxels_bool: voxel filled in new bool values.
    """

    radius_table = {"C": 0.70, "N": 0.65, "O": 0.60, "S": 1.00, "H": 0.25}

    dists_element_voxels_coords = np.sqrt(np.sum((voxels_coords - atom_coord) ** 2, axis=-1))
    # (20, 20, 20)
    contact_element_voxel_coords = np.where(dists_element_voxels_coords < radius_table[atom.element])
    add_bool = contact_element_voxel_coords
    voxels_bool[add_bool[0].T, add_bool[1].T, add_bool[2].T] = True

    return voxels_bool


def visualize_voxels(voxel_list: List):
    """This function visualizes the voxels of CNOS.
    Args:
        voxel_list: a list include all voxels of 4 channels.
    """
    voxelarray = voxel_list[0] | voxel_list[1] | voxel_list[2] | voxel_list[3] | voxel_list[4]
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxel_list[0]] = "green"
    colors[voxel_list[1]] = "blue"
    colors[voxel_list[2]] = "red"
    colors[voxel_list[3]] = "yellow"
    colors[voxel_list[4]] = "purple"
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()


def main():
    """The main function of generating voxels."""
    # 0. Load protein structure
    pdb_name = "3gbn"
    pdb_path = "3gbn.pdb"
    struct = load_protein(pdb_name, pdb_path)

    # 1. generate atom lists for 20*20*20 voxels
    voxel_atom_lists, rot_mats, central_atom_coords = generate_voxel_atom_lists(struct)  # (num_ca, num_atoms_in_voxel)

    # 2. take one voxel as an example.
    example_index = 0
    (
        voxel_atom_list, rot_mat, central_atom_coord
    ) = (voxel_atom_lists[example_index], rot_mats[example_index], central_atom_coords[example_index])

    # 3. iterate through ["C", "N", "O", "S", "H"]
    elements = ["C", "N", "O", "S", "H"]

    all_voxel = []
    for element in elements:
        selected_element_voxel = generate_selected_element_voxel(element, voxel_atom_list, rot_mat, central_atom_coord)
        all_voxel.append(selected_element_voxel)

    # 4. visualization
    visualize_voxels(all_voxel)


if __name__ == "__main__":
    main()