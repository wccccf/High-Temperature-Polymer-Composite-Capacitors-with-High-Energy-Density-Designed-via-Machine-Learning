from copy import deepcopy
from typing import List, Tuple

import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol, GetPeriodicTable


BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}


def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def mol_to_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, allHsExplicit=True)

