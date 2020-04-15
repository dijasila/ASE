from ase.calculators.eam import EAM
from ase.io.vasp import read_vasp
import numpy as np

al2cu = read_vasp('./POSCAR')
al2cu.set_calculator(EAM(potential='./AlCu.adp'))
print(al2cu.get_potential_energy())
