"""SAFIRES Tutorial input script

Ar Lennard Jones parameters obtained from 10.1103/PhysRev.136.A405.

For more information on SAFIRES, refer to the original publication:
DOI 10.1021/acs.jctc.1c00522

@author: bjk24
date: 2022-05-06
"""
import numpy as np
from operator import itemgetter

from ase import Atoms
from ase import units
from ase.calculators.lj import LennardJones
from ase.md import Langevin
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from ase.md.safires import SAFIRES

# Set up and pre-equilibrate model system from scratch.
atoms = Atoms('Ar', positions=[[0, 0, 0]])
vol = (atoms.get_masses()[0] / units._Nav / (1.374 / 1e24))**(1 / 3.)
atoms.cell = [vol, vol, vol]
atoms.center()
atoms = atoms.repeat((8, 8, 8))
atoms.pbc = [True, True, True]

atoms.calc = LennardJones(epsilon=120 * units.kB, sigma=3.4)

md = Langevin(atoms, timestep=1 * units.fs, temperature_K=94.4, 
              friction=0.01, logfile='preopt.log')
traj_preopt = Trajectory('preopt.traj', 'w', atoms)
md.attach(traj_preopt.write, interval=100)

# Run NVT preopt run.
md.run(2000)

# Prepare model for SAFIRES.
# Expand solute.
atoms.wrap()
atoms.set_tags(3)
center = atoms.cell.diagonal() / 2
distances = [[np.linalg.norm(atom.position - center), atom.index]
             for atom in atoms]
index_c = sorted(distances, key=itemgetter(0))[0][1]
atoms[index_c].tag = 1

# Expand inner region.
ninner = int(len(atoms) * 0.05) + 1  # + 1 for the solute
distances = [[atoms.get_distance(index_c, atom.index, mic=True), atom.index]
             for atom in atoms]
distances = sorted(distances, key=itemgetter(0))
for i in range(ninner + 1):
    # Start counting from i + 1 to ignore the solute, which
    # is on top of this list with a distance of zero.
    atoms[distances[i+1][1]].tag = 2

# Constrain the central particle (origin).
atoms.constraints = [FixAtoms(indices=[index_c])]

# Set up SAFIRES dynamics.
md = SAFIRES(atoms, timestep=1 * units.fs, friction=0,
             temperature_K=0, natoms=1, logfile="md.log")
traj = Trajectory('md.traj', 'w', atoms)
md.attach(traj.write, interval=1)

# Run NVE SAFIRES.
md.run(1000)
