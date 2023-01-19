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

num_of_voxels = 40
len_of_voxel = 0.4


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


def generate_vertices(central_coord: np.array, rot: np.array) -> np.array:
    """This function generates rotated coordinates of vertices of box
        according to the coordinate of the central position.

    Args:
        central_coord: coordinate of the central position of the box.
        rot: rotation matrix that rotation corner vector to ca_cb direction.

    Returns:
        vetices_coords: coordinates of 8 vertices.
    """
    left_down_corner = central_coord - num_of_voxels / 2 * len_of_voxel
    factor_list = np.array(list(product("01", repeat=3))).astype(np.int32)
    vertices_coords = factor_list * num_of_voxels * len_of_voxel + left_down_corner
    vertices_centre_vector = vertices_coords - central_coord
    # vector addition, (8, 3)
    vertices_coords = (vertices_centre_vector@rot.T) + central_coord
    return vertices_coords


def is_in_range(vertices_coords: np.array, atom: Bio.PDB.Atom.Atom, center_coord: np.array) -> bool:
    """This function determines whether the given atom is inside the range of the box.

    Args:
        vertices_coords: rotated coordinates of 8 vertices.
        atom: selected atom.
        center_coord: centre coordinate of the box
    """
    length_of_box = num_of_voxels * len_of_voxel
    atom_center_vector = atom.coord - center_coord
    vector_coord_diff = vertices_coords - vertices_coords[0]
    x_vector, y_vector, z_vector = vector_coord_diff[1], vector_coord_diff[2], vector_coord_diff[4]
    x_vector = x_vector / np.linalg.norm(x_vector)
    y_vector = y_vector / np.linalg.norm(y_vector)
    z_vector = z_vector / np.linalg.norm(z_vector)

    in_range = (abs(atom_center_vector @ x_vector) * 2 < length_of_box) and \
               (abs(atom_center_vector @ y_vector) * 2 < length_of_box) and \
               (abs(atom_center_vector @ z_vector) * 2 < length_of_box)

    # xyz_mat = np.array([x_vector, y_vector, z_vector])
    # xyz_mat = xyz_mat / np.linalg.norm(xyz_mat, axis=1)
    # in_range = np.all(xyz_mat@atom_center_vector * 2 < length_of_box)

    return in_range


def select_in_range_atoms(
        struct, ca_list, cb_list,
        selected_central_atom_type="CA", shift=0
) -> Tuple[List, List, np.array]:
    """This function generates the selected in-range atom corresponding to the central atom ["CA", "CB"].

    Args:
        struct: structure in biopython.
        ca_list: list containing CA in the PDB file.
        cb_list: list containing CB in the PDB file.
        selected_central_atom_type: string stating which central atom is selected.
        shift: shift of carbon alpha along the diagonal direction.

    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        central_atom_coords: list of central atom coordinate after applying shift.
        vertices_coords: np.array of 8 vertices coordinates of voxels. (num_of_res, (8, 3))
    """
    central_atom_list = ca_list if selected_central_atom_type == "CA" else cb_list

    # get ca_cb vector and corner vector
    ca_cb_vectors = [Vector(ca.get_coord()) - Vector(cb.get_coord()) for ca, cb in zip(ca_list, cb_list)]
    corner_vector = Vector(1, 1, 1)

    voxel_atom_lists = []

    # search atoms that are within the range
    # This function takes time to initialize, so only initialize once outside the loop, the coordinates
    # of the atoms should not be transformed but only using rotation matrix to left-multiply.
    nb_search = NeighborSearch(list(struct.get_atoms()))

    # record rotated coordinates of vertices
    vertices_coords = []

    # record central atom coordinates
    central_atom_coords = []

    # searching for atoms that are within the range
    for central_atom, ca_cb_vector in tqdm(zip(central_atom_list, ca_cb_vectors), total=len(ca_cb_vectors)):
        # the first atom in the voxel is the central atom.
        voxel_atom_list = [central_atom]

        # generate rotation matrix, rotate corner_vector to ca_cb_vector
        rot = np.identity(3) if selected_central_atom_type == "CA" else rotmat(corner_vector, ca_cb_vector)

        # Shift of Ca position can be applied by confirming the shift of the central atom, i.e. by confirming the new
        # central atom coordinates, vector subtraction
        central_atom_coord = central_atom.get_coord() - shift * ca_cb_vector.normalized().get_array()

        # generate eight rotated box vertices, (8, 3)
        vertices_coord = generate_vertices(central_atom_coord, rot)

        # search in-sphere atoms
        # It doesn't matter if not transferring atom coordinate here because sphere is invariant.
        searched_atom_list = nb_search.search(central_atom_coord, np.sqrt(3 * 10 ** 2) * 0.8)

        # keep in-range atoms
        for atom in searched_atom_list:
            # determine whether the selected atom is in range of the rotated box.
            bool_is_in_range = is_in_range(vertices_coord, atom, central_atom_coord)
            if bool_is_in_range and atom != central_atom and atom.parent != central_atom.parent:
                voxel_atom_list.append(atom)
            if atom.parent == central_atom.parent and not bool_is_in_range:
                raise ValueError("Shift value needs to be justified,"
                                 "atoms from central residual is not included in the box! "
                                 "Please use a different shift value.")
        # record atoms that are within the range
        voxel_atom_lists.append(voxel_atom_list)

        # save central atom coordinate
        central_atom_coords.append(central_atom_coord)

        # save vertices coordinates for filling the atom in voxels.
        vertices_coords.append(vertices_coord)

    return voxel_atom_lists, central_atom_coords, np.array(vertices_coords)


def generate_voxel_atom_lists(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List, np.array]:
    """This function generates permitted range for every selected CA. (20 * 20 * 20), 0.8 A for dimension.

    Args:
        struct: structure in biopython.

    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        central_atom_coords: list of central atom coordinate after applying shift.
        vertices_coords: np.array of 8 vertices coordinates of voxels. (num_of_res, (8, 3))
    """
    ca_list, cb_list = generate_central_atoms(struct)
    voxel_atom_lists, central_atom_coords, vertices_coords = select_in_range_atoms(
        struct, ca_list, cb_list, selected_central_atom_type="CB", shift=0
    )
    return voxel_atom_lists, central_atom_coords, vertices_coords


def generate_cubic_centre_coords(vertices_coord: np.array) -> np.array:
    """This function generates centric coordinates for each of the sub-voxel, by given 8 coordinates of big voxel.

    Args:
        vertices_coord: coordinates of 8 vertices

    Returns:
        sub_voxel_centre_coords: sub voxel center coordinates
    """
    corner_vector = Vector(1, 1, 1)
    rotated_corner_vector = Vector(vertices_coord[-1] - vertices_coord[0])
    # define rotation matrix that rotate the corner vector back to standard xyz system
    rot = rotmat(rotated_corner_vector, corner_vector)
    # define the inverse rotation matrix
    rot_inv = rotmat(corner_vector, rotated_corner_vector)

    # rotate to standard xyz system
    cartesian_vertices_coord = (vertices_coord - vertices_coord[0])@rot.T + vertices_coord[0]
    left_down_corner_coord, right_up_corner_coord = cartesian_vertices_coord[0], cartesian_vertices_coord[-1]
    x_split = np.linspace(left_down_corner_coord[0], right_up_corner_coord[0], num=num_of_voxels + 1)[
              :-1] + (right_up_corner_coord[0] - left_down_corner_coord[0]) / num_of_voxels / 2
    y_split = np.linspace(left_down_corner_coord[1], right_up_corner_coord[1], num=num_of_voxels + 1)[
              :-1] + (right_up_corner_coord[1] - left_down_corner_coord[1]) / num_of_voxels / 2
    z_split = np.linspace(left_down_corner_coord[2], right_up_corner_coord[2], num=num_of_voxels + 1)[
              :-1] + (right_up_corner_coord[2] - left_down_corner_coord[2]) / num_of_voxels / 2
    # centre coordinates of sub voxels (20, 20, 20, 3)
    sub_voxel_centre_coords = np.array([[[[x_split[i], y_split[j], z_split[k]]
                                          for i in range(num_of_voxels)]
                                         for j in range(num_of_voxels)]
                                        for k in range(num_of_voxels)])

    # rotate to original angle
    sub_voxel_centre_coords = (sub_voxel_centre_coords - vertices_coord[0])@rot_inv.T + vertices_coord[0]
    return sub_voxel_centre_coords


def generate_selected_element_voxel(
        selected_element: str, voxel_atom_list: List, central_atom_coord: np.array, vertices_coord: np.array,
):
    """This function generate selected-element voxel by inserting True/False to 20*20*20 voxel.

    Args:
        selected_element: selected element for one single channel, "C", "N", "O", "S".
        voxel_atom_list: The pre-generated voxel list containing central CA atoms and the surrounding atoms.
        central_atom_coord: list of central atom coordinate after applying shift.
        vertices_coord: coordinates of 8 vertices of the box for the selected residue.
    """
    if selected_element not in ["C", "N", "O", "S"]:
        raise ValueError("'selected_element' has to be in the options of 'C', 'N', 'S', 'O'")

    # 1. create True False voxel.
    voxels_bool = (np.ones((num_of_voxels, num_of_voxels, num_of_voxels)) == 0).astype(bool)

    # 2. selected corresponding atoms
    selected_atom_list = []
    for atom in voxel_atom_list:
        if atom.element == selected_element and atom.element in ["C", "N", "O", "S"]:
            selected_atom_list.append(atom)

    # 3. Generate the coordinates of 20 * 20 * 20 voxels,
    # one centric coordinate for one sub voxel, (20, 20, 20, 3)
    sub_voxel_centre_coords = generate_cubic_centre_coords(vertices_coord)

    # 4. Add atom to the voxel.
    for atom in selected_atom_list:
        voxels_bool = add_atom_to_voxel(atom, atom.coord, sub_voxel_centre_coords, voxels_bool)
    return voxels_bool


def add_atom_to_voxel(
        atom: Bio.PDB.Atom.Atom,
        atom_coord: np.array,
        sub_voxel_centre_coords: ndarray,
        voxels_bool: ndarray,
) -> ndarray:
    """This function adds atom to the corresponding location of the voxel.

    Args:
        atom: atom object to be added
        atom_coord: atom coordinate of atom to be added
        sub_voxel_centre_coords: coordinates of the 20*20*20 voxels.
        voxels_bool: voxel array in bool.

    Returns:
        voxels_bool: voxel filled in new bool values.
    """

    radius_table = {"C": 0.70, "N": 0.65, "O": 0.60, "S": 1.00}
    dists_element_voxels_coords = np.sqrt(np.sum((sub_voxel_centre_coords - atom_coord) ** 2, axis=-1))
    # (20, 20, 20)
    add_bool = np.where(dists_element_voxels_coords < radius_table[atom.element])
    # print("dist coord", sub_voxel_centre_coords.reshape(-1, 3)[np.argmin(dists_element_voxels_coords)])
    # print("atom:", atom.coord)
    # print(np.min(dists_element_voxels_coords))
    voxels_bool[add_bool[0].T, add_bool[1].T, add_bool[2].T] = True

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
    voxel_atom_lists, central_atom_coords, vertices_coords = generate_voxel_atom_lists(struct)

    # 2. take one voxel as an example.
    example_index = -1
    (
        voxel_atom_list, central_atom_coord, vertices_coord
    ) = (voxel_atom_lists[example_index], central_atom_coords[example_index], vertices_coords[example_index])
    # 3. iterate through ["C", "N", "O", "S"]
    elements = ["C", "N", "O", "S"]
    all_voxel = []
    for element in elements:
        selected_element_voxel = generate_selected_element_voxel(
            element, voxel_atom_list, central_atom_coord, vertices_coord
        )
        all_voxel.append(selected_element_voxel)

    # 4. visualization
    visualize_voxels(all_voxel)


if __name__ == "__main__":
    main()
