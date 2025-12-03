# utils/graph_builder.py
import torch
from torch_geometric.data import Data

import numpy as np

from pymatgen.core import Structure

import ase
from ase.neighborlist import neighbor_list
from ase.io import read


def get_host_atoms(filename, am='Li'):

    """
    Reads a POSCAR file with structure
    Returns 
    """

    atoms = read(filename, format='vasp')
    indices_to_remove = [atom.index for atom in atoms if atom.symbol == am]

    del atoms[indices_to_remove]

    return atoms

def get_last_am_coordinates(atoms, am='Li'):
    """
    Returns alkali metal coordinates of the last added AM (with the highest index)
    """
    am_indices = [atom.index for atom in atoms if atom.symbol == am]
    
    if not am_indices:
        raise ValueError(f"no {am} in structure")

    last_am_index = max(am_indices)
    last_am_position = atoms[last_am_index].position
    
    return last_am_position

def build_graph_with_am(
    host_structure,       # host structure (without alkali metal)
    am = 'Li',            # alkali metal in the structure
    am_position=None,     # am coordinates (a 3D vector)
    ienergy=None,         # intercalation energy (optional)
    cutoff=6.0,           # cut-off radius for the graph edges
):
    """
    Builds a crystal graph with noted alkali metal (AM) atoms
    Returns a Data Object for PyTorch Geometric
    """
    if am_position != None:
      # create a structure with AM
      structure_with_am = host_structure.copy()
      structure_with_am.extend(Atoms(am, positions=[am_position]))

    else:
      am_position = get_last_am_coordinates(host_structure)
      structure_with_am = host_structure.copy()
    
    # get information on neighbours
    i, j, d, D = neighbor_list('ijdD', structure_with_am, cutoff)
    
    # create nodes
    node_features = []
    for atom in structure_with_am:
        features = [
            atom.number,
            atom.mass,
            1.0 if atom.symbol == am else 0.0,  # put a flag on AM atom
        ]
        node_features.append(features)
    
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # create edges
    edge_index = torch.tensor([i, j], dtype=torch.long)
    
    # edge features (distance, vector)
    edge_features = torch.tensor(D, dtype=torch.float)
    
    # create Data object
    if am_position != None:
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=torch.tensor([ienergy], dtype=torch.float) if ienergy else None,
            am_position=torch.tensor(am_position, dtype=torch.float),
            host_size=len(host_structure) # add if-else depending on was AM added manually or not
        )

    else:
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=torch.tensor([ienergy], dtype=torch.float) if ienergy else None,
            am_position=torch.tensor(am_position, dtype=torch.float),
            host_size=len(host_structure)-1 
        )
    
    return graph



