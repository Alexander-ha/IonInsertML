from ase import Atoms

host = Atoms(symbols=symbols,
             positions=positions,
             cell=rprimd,
             pbc=True)