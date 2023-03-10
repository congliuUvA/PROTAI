"""This module is for voxel box generation."""
import Bio
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import List, Tuple
from Bio.PDB.NeighborSearch import NeighborSearch
from itertools import product
from Bio.PDB.vectors import Vector, rotmat, rotaxis2m
from Bio.PDB.PDBParser import PDBParser
from pathlib import Path
from voxel_rotate_atom import gen_ca_cb_vectors, visualize_voxels, cal_sasa
import freesasa
import hydra

num_of_voxels = 20
len_of_voxel = 0.8


def load_protein(arguments, pdb_name: str, file_path: str) -> Bio.PDB.Structure.Structure:
    """This function is used to load the protein file in format of mmCIF to the structure in Biopython.

    Args:
        arguments: arguments input by users.
        pdb_name: PDB ID
        file_path: file path of the protein file.

    Returns:
        struct: Structure of the selected protein in Biopython.
    """

    struct = PDBParser(QUIET=1).get_structure(pdb_name, file_path)

    # remove hetero residues from struct
    for model in struct.get_list():
        for chain in model.get_list():
            for res in chain.get_list():
                # remove hetero residues
                if res.get_id()[0] not in [" ", "W"]:
                    chain.__delitem__(res.get_id())

    # reset atomic serial number for getting access to sasa results
    idx = -1
    for atom in struct.get_atoms():
        atom.set_serial_number(idx)

    return struct


def generate_vertices(central_coord: np.array, rot: np.array, n_coord, ca_cb_vector,):
    """This function generates rotated coordinates of vertices of box
        according to the coordinate of the central position.

    Args:
        central_coord: coordinate of the central position of the box.
        rot: rotation matrix that rotation corner vector to ca_cb direction.
        n_coord: coordinates of atom N.
        ca_cb_vector: ca_cb vector.

    Returns:
        vetices_coords: coordinates of 8 vertices.
    """
    left_down_corner = central_coord - num_of_voxels / 2 * len_of_voxel
    factor_list = np.array(list(product("01", repeat=3))).astype(np.int32)
    vertices_coords = factor_list * num_of_voxels * len_of_voxel + left_down_corner
    vertices_centre_vector = vertices_coords - central_coord
    # vector addition, (8, 3)
    vertices_coords = (vertices_centre_vector@rot.T) + central_coord
    # make sure each orientation is the same.
    left_up_corner = vertices_coords[3]
    vector_cblc = left_up_corner - central_coord
    vector_cbn = n_coord - central_coord
    oro_vector_cacbn_plane = np.cross(vector_cbn, ca_cb_vector.get_array())
    oro_vector_lccacb_plane = np.cross(ca_cb_vector.get_array(), vector_cblc)
    angle_between_plane = (oro_vector_cacbn_plane @ oro_vector_lccacb_plane) / \
                          (np.linalg.norm(oro_vector_cacbn_plane) * np.linalg.norm(oro_vector_lccacb_plane))
    # angle_between_plane = min(max(angle_between_plane, -1), 1)
    angle = np.arccos(angle_between_plane)
    clock_wise = ca_cb_vector.get_array() @ np.cross(oro_vector_cacbn_plane, oro_vector_lccacb_plane) > 0
    rot_angle = 1 + angle if not clock_wise else 1 - angle
    rot_further = rotaxis2m(rot_angle, ca_cb_vector)
    vertices_centre_vector = vertices_coords - central_coord
    vertices_coords = (vertices_centre_vector @ rot_further.T) + central_coord

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

    return in_range


def select_in_range_atoms(
        struct, ca_list, cb_list, ca_cb_vectors, n_coords,
        selected_central_atom_type="CA", shift=0
) -> Tuple[List, List, np.array]:
    """This function generates the selected in-range atom corresponding to the central atom ["CA", "CB"].

    Args:
        struct: structure in biopython.
        ca_list: list containing CA in the PDB file.
        cb_list: list containing CB in the PDB file.
        ca_cb_vectors: list containing ca-cb vectors.
        n_coords: coordinates of Nitrogen.
        selected_central_atom_type: string stating which central atom is selected.
        shift: shift of carbon alpha along the diagonal direction.

    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        central_atom_coords: list of central atom coordinate after applying shift.
        vertices_coords: np.array of 8 vertices coordinates of voxels. (num_of_res, (8, 3))
    """
    central_atom_list = ca_list if selected_central_atom_type == "CA" else cb_list

    # get ca_cb vector and corner vector
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
    for central_atom, ca_cb_vector, n_coord in tqdm(
            zip(central_atom_list, ca_cb_vectors, n_coords), total=len(ca_cb_vectors)
    ):
        # the first atom in the voxel is the central atom.
        voxel_atom_list = [central_atom]

        # generate rotation matrix, rotate corner_vector to ca_cb_vector
        rot = np.identity(3) if selected_central_atom_type == "CA" else rotmat(corner_vector, ca_cb_vector)

        # Shift of Ca position can be applied by confirming the shift of the central atom, i.e. by confirming the new
        # central atom coordinates, vector subtraction
        central_atom_coord = central_atom.get_coord() - shift * ca_cb_vector.normalized().get_array()

        # generate eight rotated box vertices, (8, 3)
        vertices_coord = generate_vertices(central_atom_coord, rot, n_coord, ca_cb_vector)

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


    return voxel_atom_lists, central_atom_coords, np.array(vertices_coords),


def generate_voxel_atom_lists(struct: Bio.PDB.Structure.Structure) -> Tuple[List, List, np.array]:
    """This function generates permitted range for every selected CA. (20 * 20 * 20), 0.8 A for dimension.

    Args:
        struct: structure in biopython.

    Returns:
        voxel_atom_lists: list of atoms contained in each 20*20*20 voxels.
        central_atom_coords: list of central atom coordinate after applying shift.
        vertices_coords: ndarray of 8 vertices coordinates of voxels. (num_of_res, (8, 3))
    """
    ca_list, cb_list, ca_cb_vectors, boxes_counter, n_coords = gen_ca_cb_vectors(struct)
    voxel_atom_lists, central_atom_coords, vertices_coords, = select_in_range_atoms(
        struct, ca_list, cb_list, ca_cb_vectors, n_coords, selected_central_atom_type="CB", shift=0
    )
    return voxel_atom_lists, central_atom_coords, vertices_coords,


def generate_cubic_centre_coords(vertices_coord: np.array) -> np.array:
    """This function generates centric coordinates for each of the sub-voxel, by given 8 coordinates of big voxel.

    Args:
        vertices_coord: coordinates of 8 vertices

    Returns:
        sub_voxel_centre_coords: sub voxel center coordinates
    """
    x_vector, y_vector, z_vector = \
        vertices_coord[1] - vertices_coord[0], \
        vertices_coord[2] - vertices_coord[0], \
        vertices_coord[4] - vertices_coord[0]
    x_vector, y_vector, z_vector = \
        x_vector / np.linalg.norm(x_vector), \
        y_vector / np.linalg.norm(y_vector), \
        z_vector / np.linalg.norm(z_vector)
    sub_voxel_centre_coords = []
    for i in range(num_of_voxels):
        for j in range(num_of_voxels):
            for k in range(num_of_voxels):
                sub_centre = vertices_coord[0] + \
                x_vector * (i * len_of_voxel + len_of_voxel / 2) + \
                y_vector * (j * len_of_voxel + len_of_voxel / 2) + \
                z_vector * (k * len_of_voxel + len_of_voxel / 2)
                sub_voxel_centre_coords.append(sub_centre)
    sub_voxel_centre_coords = np.reshape(np.array(sub_voxel_centre_coords), (20, 20, 20, 3))
    return sub_voxel_centre_coords


def generate_selected_element_voxel(
        arguments, elements_list: List, selected_element: str, voxel_atom_list: List,
        vertices_coord: np.array, sasa_results: freesasa.Result,
):
    """This function generate selected-element voxel by inserting True/False to 20*20*20 voxel.

    Args:
        arguments: arguments of the user.
        elements_list: elements that are in interets.
        selected_element: selected element for one single channel, "C", "N", "O", "S".
        voxel_atom_list: The pre-generated voxel list containing central CA atoms and the surrounding atoms.
        vertices_coord: coordinates of 8 vertices of the box for the selected residue.
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
    for atom in voxel_atom_list[1:]:
        if atom.element == selected_element and atom.element in elements_list:
            selected_atom_list.append(atom)

    # 3. Generate the coordinates of 20 * 20 * 20 voxels,
    # one centric coordinate for one sub voxel, (20, 20, 20, 3)
    sub_voxel_centre_coords = generate_cubic_centre_coords(vertices_coord)

    # 4. Add atom to the voxel.
    for atom in selected_atom_list:
        voxels_bool, partial_charges, sasa = add_atom_to_voxel(
            arguments, atom, atom.coord, sub_voxel_centre_coords, voxels_bool, partial_charges, sasa_results, sasa,
        )
    return voxels_bool, partial_charges, sasa


def add_atom_to_voxel(
        arguments,
        atom: Bio.PDB.Atom.Atom,
        atom_coord: np.array,
        sub_voxel_centre_coords: ndarray,
        voxels_bool: ndarray,
        partial_charges: ndarray,
        sasa_results: freesasa.Result,
        sasa: ndarray,
) -> Tuple[ndarray, ndarray, ndarray]:
    """This function adds atom to the corresponding location of the voxel.

    Args:
        arguments: arguments of the user.
        atom: atom object to be added.
        atom_coord: atom coordinate of atom to be added.
        sub_voxel_centre_coords: coordinates of the 20*20*20 voxels.
        voxels_bool: voxel array in bool.
        partial_charges: partial_charge channel, 20*20*20 array.
        sasa_results: results of solvent accessible surface area.
        sasa: solvent accessible surface area array.

    Returns:
        voxels_bool: voxel filled in new bool values.
        partial_charges: filled partial charge channel, 20*20*20 array.
        sasa: solvent accessible surface area array.
    """

    radius_table = {"C": 1.7, "O": 1.37, "N": 1.45, "P": 1.49, "S": 1.7, "H": 1.0}
    dists_element_voxels_coords = np.sqrt(np.sum((sub_voxel_centre_coords - atom_coord) ** 2, axis=-1))
    # (20, 20, 20)
    add_bool = np.where(dists_element_voxels_coords < radius_table[atom.element])
    voxels_bool[add_bool[0].T, add_bool[1].T, add_bool[2].T] = True
    if arguments.add_partial_charges:
        partial_charges[add_bool[0].T, add_bool[1].T, add_bool[2].T] = atom.occupancy
    if arguments.add_sasa:
        sasa[add_bool[0].T, add_bool[1].T, add_bool[2].T] = sasa_results.atomArea(atom.get_serial_number()-1)

    return voxels_bool, partial_charges, sasa


@hydra.main(version_base=None, config_path="../../config/voxel_box", config_name="voxel_box")
def main(arguments):
    """The main function of generating voxels."""
    # 0. Load protein structure
    pdb_name = "3c70"
    pdb_path = "3c70.pdb1"
    struct = load_protein(arguments, pdb_name, pdb_path)

    # 1. generate atom lists for 20*20*20 voxels
    voxel_atom_lists, central_atom_coords, vertices_coords = generate_voxel_atom_lists(struct)

    # 2. take one voxel as an example.
    example_index = 0
    (
        voxel_atom_list, central_atom_coord, vertices_coord
    ) = (voxel_atom_lists[example_index], central_atom_coords[example_index], vertices_coords[example_index])
    # 3. iterate through ["C", "N", "O", "S", "H"]
    elements_list = ["C", "N", "O", "S", "H"] if arguments.addH else ["C", "N", "O", "S"]

    # calculate sasa results
    sasa_results = cal_sasa(struct, pdb_name) if arguments.add_sasa else None

    all_voxel = []
    all_partial_charges = []
    all_sasa = []
    for element in elements_list:
        selected_element_voxel, partial_charges, sasa = generate_selected_element_voxel(
            arguments,
            elements_list,
            element,
            voxel_atom_list,
            vertices_coord,
            sasa_results
        )
        all_voxel.append(selected_element_voxel)
        all_partial_charges.append(partial_charges)
        all_sasa.append(sasa)

    # 4. visualization
    visualize_voxels(arguments, all_voxel)

    # 5. save box coordinates
    box_coords_dir_path = Path.cwd().parent.joinpath("box_coords")
    if not box_coords_dir_path.exists():
        box_coords_dir_path.mkdir()
    structure_box_coords_path = box_coords_dir_path.joinpath(Path(pdb_name).stem + ".npy")
    print(f"save coord at {str(structure_box_coords_path)}")
    np.save(str(structure_box_coords_path), vertices_coords)


if __name__ == "__main__":
    main()
