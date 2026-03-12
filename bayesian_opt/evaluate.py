import numpy as np 
import pandas as pd
import logging 
import argparse

from ase import Atoms
from ase.io import read
from ase.calculators.vasp import Vasp

from bayesian_opt.bo import BayesianOptimization
from utils.data_loader import load_train, load_host


"""body of the script"""

"""
In run file set paths and initials:

train_file_csv = './training_Li1.csv'
host_file = './POSCAR_host'
host_energy = 
int_atom = 'Li'
int_atom_energy = 
batch_size = 
strategy = 'constant_liar'

"""

host_atoms = load_host(host_file)
host_positions = host_atoms.get_positions()
rprimd = host_atoms.get_cell()[:]

X_init, y_init = load_train(train_file_csv)

k = X_init.shape[1] // 3 

bo = BayesianOptimization(
    int_atom=int_atom,
    n_candidates=500,
    rmin=1.45,
    rmin_insrt=1.34,
    host=host_positions,
    rprimd=rprimd,
    random_state=42         
)

bo.fit(X_init, y_init)
X_new = bo.suggest(batch_size=batch_size, xi=0.01, strategy=strategy)

for i in range(X_new.shape[0]):
    atoms = host_atoms.append(int_atom, positions=[X_new[i]])



def compute_y(total_energy, host_energy, int_atom_energy, n_int_atoms):
    """intercalation energy computation for a new configuration suggested by BO"""

    return (total_energy - host_energy - int_atom_energy) / n_int_atoms






