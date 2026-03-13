import numpy as np 
import pandas as pd
import os 

from ase import Atom
from ase import Atoms
from ase.io import read
from ase.io.vasp import read_incar
from ase.calculators.vasp import Vasp

from bayesian_opt.bo import BayesianOptimization
from utils.data_loader import load_train, load_host



"""
In run file set paths and initials:

train_file_csv = './training_Li1.csv'

host_file = './POSCAR_host'

params_file = './INCAR'
    (VASP calculation parameters)

host_energy = 
    (host energy for y_new calculations)

int_atom = 'Li' 
    (type of atom insterted in the host structure)

int_atom_energy =  
    (reference energy for y_new calculations)

batch_size = 3 
    (how many new suggested points will be calculated at the same time)

strategy = 'constant_liar' 
    (how too choose points for batch formation)

N = 100
    (number of active learning iterations)
"""


def compute_y(total_energy, host_energy, int_atom_energy, n_int_atoms):
    """intercalation energy computation for a new configuration suggested by BO"""

    return (total_energy - host_energy - int_atom_energy) / n_int_atoms


"""body of the script"""

params = read_incar(params_file)

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

X_train, y_train = X_init.copy(), y_init.copy()

for _ in range(N):

    bo.fit(X_train, y_train)
    X_new = bo.suggest(batch_size=batch_size, xi=0.01, strategy=strategy)

    for i in range(X_new.shape[0]):

        atoms = host_atoms.copy()
        atoms.append(Atom(int_atom, position=position))

        calc_dir = f'calc_{idx}'
        os.makedirs(calc_dir, exist_ok=True)

        calc = Vasp(directory=calc_dir, **params)
        atoms.calc = calc

        # run VASP
        try:
            total_energy = atoms.get_potential_energy()
        except Exception as e:
            logging.error(f"Calculation for {idx} failed: {e}")
            continue

        # compute new point's y (intercalation energy)
        y_new = compute_y(total_energy, host_energy, int_atom_energy, n_int_atoms=1)

        # update training set
        X_train = np.vstack([X_train, position.reshape(1, -1)])
        y_train = np.append(y_train, y_new)












