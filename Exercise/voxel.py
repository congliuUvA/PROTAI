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


def range_editor(coord: ndarray) -> List:
    """This function edits the range of the coordinates,
    given the central position of the ca atom under the cartesian coordinates.

    Args:
        coord: central ca coords.

    Returns:
        range_list: list of x,y,z range
    """
    x, y, z = coord[0], coord[1], coord[2]
    range_x = [x - 10 * 0.8, x + 10 * 0.8]
    range_y = [y - 10 * 0.8, y + 10 * 0.8]
    range_z = [z - 10 * 0.8, z + 10 * 0.8]
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

    return ca_list, cb_list


def select_in_range_atoms(struct, ca_list, cb_list, selected_central_atom_type="CA") -> Tuple[List, List]:
    """This function generates the selected in-range atom corresponding to the central atom ["CA", "CB"].

    Args:
        struct: structure in biopython.
        ca_list: list containing CA in the PDB file.
        cb_list: list containing CB in the PDB file.
        selected_central_atom_type: string stating which central atom is selected.

    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        rot_mat: list of rotation matrix corresponding to every residue.
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

    for central_atom, ca_cb_vector in tqdm(zip(central_atom_list, ca_cb_vectors), total=len(ca_cb_vectors)):
        # the first atom in the voxel is the central atom.
        voxel_atom_list = [central_atom]

        # generate rotation matrix
        rot = np.identity(3) if selected_central_atom_type == "CA" else rotmat(corner_vector, ca_cb_vector)

        # search in-sphere atoms
        # It doesn't matter if not transferring atom coordinate here but sphere is invariant.
        searched_atom_list = nb_search.search(central_atom.get_coord(), np.sqrt(3*10**2) * 0.8)

        # keep in-range atoms
        for atom in searched_atom_list:
            # remember to left-multiply rotation matrix all the time when requiring atom coordinates.
            central_range_list = range_editor(rot@central_atom.get_coord())
            atom_coord = rot@atom.get_coord()
            range_x, range_y, range_z = central_range_list[0], central_range_list[1], central_range_list[2]
            atom_x, atom_y, atom_z = atom_coord[0], atom_coord[1], atom_coord[2]
            if range_x[0] < atom_x < range_x[1] and \
                    range_y[0] < atom_y < range_y[1] and \
                    range_z[0] < atom_z < range_z[1]:
                if atom != central_atom and atom.parent != central_atom.parent:
                    voxel_atom_list.append(atom)
        voxel_atom_lists.append(voxel_atom_list)

        # save rotation matrix
        rot_mats.append(rot)

    return voxel_atom_lists, rot_mats


def generate_voxel_atom_lists(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List]:
    """This function generates permitted range for every selected CA. (20 * 20 * 20), 0.8 A for dimension.

    Args:
        struct: structure in biopython.
    """
    ca_list, cb_list = generate_central_atoms(struct)
    voxel_atom_lists, rot_mats = select_in_range_atoms(struct, ca_list, cb_list, selected_central_atom_type="CB")
    return voxel_atom_lists, rot_mats


def generate_selected_element_voxel(selected_element: str, voxel_atom_list: List, rot_mat: np.array):
    """This function generate selected-element voxel by inserting True/False to 20*20*20 voxel.

    Args:
        selected_element: selected element for one single channel, "C", "N", "O", "S".
        voxel_atom_list: The pre-generated voxel list containing central CA atoms and the surrounding atoms.
        rot_mat: rotation matrix for the selected residue.
    """
    if selected_element not in ["C", "N", "O", "S"]:
        raise ValueError("'selected_element' has to be in the options of 'C', 'N', 'S', 'O'")

    # 1. create True False voxel.
    voxels_bool = (np.ones((20, 20, 20)) == 0).astype(bool)

    # 2. selected corresponding atoms
    selected_atom_list = [voxel_atom_list[0]]
    for atom in voxel_atom_list[1:]:
        if atom.element == selected_element and atom.element in ["C", "N", "O", "S"]:
            selected_atom_list.append(atom)

    # 3. Generate the coordinates of 20 * 20 * 20 voxels, (20, 20, 20, 8, 3)
    initial_atom_coord = rot_mat@selected_atom_list[0].coord - 10 * 0.8
    voxels_coords = [[[gen_one_voxel_coords(
        np.array([[initial_atom_coord[0] + i * 0.8],
                 [initial_atom_coord[1] + j * 0.8],
                 [initial_atom_coord[2] + k * 0.8]])) for i in range(20)] for j in range(20)] for k in range(20)]
    voxels_coords = np.array(voxels_coords)

    # 4. Add atom to the voxel.
    for atom in selected_atom_list:
        atom_coord = rot_mat@atom.coord
        voxels_bool = add_atom_to_voxel(atom, atom_coord, voxels_coords, voxels_bool)
    return voxels_bool


def gen_one_voxel_coords(coord: np.array) -> np.array:
    """This function generates the eight coordinates
    of one single voxel given the coordinates of left corner of the voxel.
    """
    factor_list = np.array(list(product("01", repeat=3))).astype(np.int32)
    voxel_coords = factor_list * 0.8 + coord.T
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

    radius_table = {"C": 0.70, "N": 0.65, "O": 0.60, "S": 1.00}

    dists_element_voxels_coords = np.sqrt(np.sum((voxels_coords - atom_coord) ** 2, axis=-1))  # (20, 20, 20, 8)
    contact_element_voxel_coords = np.where(dists_element_voxels_coords < radius_table[atom.element])
    # only care about the previous three arrays, which represent the coordinate the of voxels, for the last array, it
    # represents the index of the vertex that has been occupied.
    add_bool = np.unique(np.array(contact_element_voxel_coords)[:-1].T, axis=0)  # (3, number_of_occupied_voxel)
    voxels_bool[add_bool.T[0], add_bool.T[1], add_bool.T[2]] = True

    return voxels_bool


def visualize_voxels(voxel_list: List):
    """This function visualizes the voxels of CNOS.

    Args:
        voxel_list: a list include all voxels of 4 channels.
    """
    voxelarray = voxel_list[0] | voxel_list[1] | voxel_list[2] | voxel_list[3]
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxel_list[0]] = "green"
    colors[voxel_list[1]] = "blue"
    colors[voxel_list[2]] = "red"
    colors[voxel_list[3]] = "yellow"
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()


def main():
    """The main function of generating voxels."""
    # 0. Load protein structure
    struct = load_protein("3gbn.pdb")

    # 1. generate atom lists for 20*20*20 voxels
    voxel_atom_lists, rot_mats = generate_voxel_atom_lists(struct)  # (num_ca, num_atoms_in_voxel)

    # 2. take one voxel as an example.
    example_index = 0
    voxel_atom_list, rot_mat = voxel_atom_lists[example_index], rot_mats[example_index]

    # 3. iterate through ["C", "N", "O", "S"]
    elements = ["C", "N", "O", "S"]

    all_voxel = []
    for element in elements:
        selected_element_voxel = generate_selected_element_voxel(element, voxel_atom_list, rot_mat)
        all_voxel.append(selected_element_voxel)

    # 4. visualization
    visualize_voxels(all_voxel)


if __name__ == "__main__":
    main()
