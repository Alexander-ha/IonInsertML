import numpy as np
import pandas as pd

import logging 

from ase import Atoms
from ase.io import read 


def load_train(train_file_csv, target_col='energy', coords_cols=None):
    """read training set"""

    df = pd.read_csv(train_file_csv)

    if coords_cols is None:
        coords_cols = [col for col in df.columns if col != target_col]

    X = df[coords_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    return X, y 


def load_host(host_file):
    """read POSCAR file with host structure to Atoms object"""

    host = read(host_file)
    if not np.any(host.pbc):
        logging.warning("PBC not turned on, turning on all directions")
        host.pbc = True

    return host



