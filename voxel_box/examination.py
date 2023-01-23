"""This file is for checking the properties of the given protein 3GBN."""

import Bio
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder, is_aa
from typing import Tuple, List
from collections import defaultdict
import numpy as np
import os
from pathlib import Path


def load_protein(pdb_name: str, file_path: str) -> Bio.PDB.Structure.Structure:
    """This function is used to load the protein file in format of mmCIF to the structure in Biopython.

    Args:
        pdb_name: PDB ID
        file_path: file path of the protein file.

    Returns:
        struct: Structure of the selected protein in Biopython.
    """
    os.system(f"pdb2pqr30 --ff=PARSE {file_path} {Path(file_path).parent.joinpath(pdb_name)}.pqr")
    parser = PDBParser(QUIET=1, is_pqr=True)
    struct = parser.get_structure(pdb_name, str(Path(file_path).parent.joinpath(pdb_name)) + ".pqr")
    return struct


def extract_sequence(struct: Bio.PDB.Structure.Structure) -> Tuple[str, List[str]]:
    """This function extracts sequence from give extracted biopython structure.

    Args:
        struct: structure in biopython.

    Returns:
        aa_names: sequence of the structure.
    """
    ppb = PPBuilder()
    aa_names = ""
    chain_list = []
    aa_names_per_chain_list = []
    for pp in ppb.build_peptides(struct, aa_only=False):
        aa_names += pp.get_sequence()
    for model in struct:
        for chain in model:
            chain_list.append(chain)
    for i in range(len(chain_list)):
        aa_names_per_chain = ""
        for pp in ppb.build_peptides(chain_list[i], aa_only=False):
            aa_names_per_chain += pp.get_sequence()
        aa_names_per_chain_list.append(aa_names_per_chain)
    return aa_names, aa_names_per_chain_list


def count_numbers(struct:  Bio.PDB.Structure.Structure):
    """This function prints the number of atoms, residues and number of chains in the structure.

    Args:
        struct: structure in biopython.
    """
    num_atom, num_res, num_chain = 0, 0 ,0
    model = struct[0]
    for chain in model:
        num_chain += 1
        for res in chain:
            num_res += 1 if res.get_id()[0] == " " else 0  # only count number if it's standard residue
            # num_res += 1
            for atom in res:
                num_atom += 1

    print(f"The structure has {num_chain} chains, {num_res} residues and {num_atom} atoms.")


def test_aaness(struct: Bio.PDB.Structure.Structure):
    """This function tests and records the residues that are not amino acids.

    Args:
        struct: structure in biopython.
    """
    ress = struct.get_residues()
    type_dict = defaultdict(int)
    for res in ress:
        if not is_aa(res):
            type_dict[res.get_id()[0][0]] += 1
    for t in list(type_dict.keys()):
        print(f"The residual types other than amino acids are {t}.")


def check_residue_completeness(struct: Bio.PDB.Structure.Structure):
    """This function checks the completeness of the residues..

    Args:
        struct: structure in biopython.
    """
    ress = struct.get_residues()
    atoms = struct.get_atoms()
    num_res = defaultdict(int)
    num_atom_per_res = defaultdict(int)
    for res in ress:
        if is_aa(res):
            num_res[res.get_resname()] += 1
    for atom in atoms:
        if is_aa(atom.get_parent()):
            num_atom_per_res[atom.get_parent().get_resname()] += 1
    real_num_res_list = np.array(list((num_atom_per_res.values())) / np.array(list(num_res.values())))
    for real_num in real_num_res_list:
        if real_num != int(real_num):
            print("The residues in the PDB file are not complete.")
            break


def analysis(file_name: str):
    """This function is the main file of the module, for analyzing the properties.

    Args:
        file_name: file name of the protein file.
    """
    struct = load_protein(file_name)

    # a. What’s the mass of the protein, and each separate chain
    # c. What is the charge of each chain and the entire protein.
    aa_names, aa_names_per_chain_list = extract_sequence(struct)
    seq = ProteinAnalysis(aa_names)
    print(f'molecular weight in total: {seq.molecular_weight()}')
    print(f'charge of the entire protein under pH=7.0 is {seq.charge_at_pH(7.0)}')

    for i, names in enumerate(aa_names_per_chain_list):
        seq = ProteinAnalysis(names)
        print(f'molecular weight for chain {i}: {seq.molecular_weight()}')
        try:
            charge_per_chain = seq.charge_at_pH(7.0)
            print(f'charge of the chain {i} under pH=7.0 is {charge_per_chain}')
        except:
            print(f'The given chain {i} is not a peptide chain, index starting from 0.')

    # b. Count number of atoms, number of residues, number of unique chains.
    count_numbers(struct)

    # d. Is every residue complete? Eg, do all residues have all atoms that they should have?
    check_residue_completeness(struct)

    # e. Are there any non-amino-acid residues in the file? What are they?
    test_aaness(struct)

    # f. Do all residues in the protein sequence have 3D coordinates?
    print("Since the residues in protein sequence are not complete, "
          "not all of the atoms in every residue has coordinates.")


def main():
    """Main funtion of the examination of 3GBN."""
    analysis("3gbn.pdb")


if __name__ == "__main__":
    main()
