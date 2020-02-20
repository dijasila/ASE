from ase.build import diamond100
from ase.calculators.dftb_new import DFTBPlus
from ase.test.testsuite import datafiles_directory
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.units import kB, Hartree

a = 5.40632280995384
atoms = diamond100('Si', (1, 1, 6), a=a, vacuum=6., orthogonal=True,
                   periodic=True)
atoms.set_constraint(FixAtoms(indices=range(4)))
atoms.rattle(0.01, seed=1)

atoms.calc = DFTBPlus(slako_dir=datafiles_directory,
                      kpts=(2, 2, 1),
                      hamiltonian=dict(
                          scc=True,
                          filling=('fermi', dict(
                              temperature=500 * kB / Hartree,
                          )),
                      ))


dyn = BFGS(atoms, logfile='-', trajectory='tmp.traj')
dyn.run(fmax=0.1)

e = atoms.get_potential_energy()
assert abs(e - -214.036907) < 1., e
