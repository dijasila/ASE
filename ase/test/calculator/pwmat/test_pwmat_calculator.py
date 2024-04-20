import os
import pytest

from ase import Atoms
from ase.io import read
from ase.calculators.pwmat import PWmat


@pytest.mark.skip(reason='Users may not compile PWmat or no GPU in test environment.')
def test_main():
    atom_config_str: str = """  12 atoms
 Lattice vector (Angstrom)
   3.190316E+00      5.525789E+00      0.000000E+00  
   -6.380631E+00     0.000000E+00      0.000000E+00  
   0.000000E+00      0.000000E+00      2.312977E+01  
 Position (normalized), move_x, move_y, move_z
  42         0.333333           0.166667           0.500000       1  1  1
  16         0.166667           0.333333           0.432343       1  1  1
  16         0.166667           0.333333           0.567657       1  1  1
  42         0.333333           0.666667           0.500000       1  1  1
  16         0.166667           0.833333           0.432343       1  1  1
  16         0.166667           0.833333           0.567657       1  1  1
  42         0.833333           0.166667           0.500000       1  1  1
  16         0.666667           0.333333           0.432343       1  1  1
  16         0.666667           0.333333           0.567657       1  1  1
  42         0.833333           0.666667           0.500000       1  1  1
  16         0.666667           0.833333           0.432343       1  1  1
  16         0.666667           0.833333           0.567657       1  1  1
"""
    with open("atom.config", 'w', encoding='utf-8') as f:
        f.write(atom_config_str)
    atoms: Atoms = read("atom.config")
    pwmat: PWmat = PWmat(atoms=atoms, job='scf', parallel=[1, 1], directory='.')
    pwmat.calculate()
    if os.path.isfile("atom.config"):
        os.remove("atom.config")
        