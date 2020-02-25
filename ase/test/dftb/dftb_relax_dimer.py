from ase import Atoms
from ase.test.testsuite import datafiles_directory
from ase.calculators.dftb_new import DFTBPlus
from ase.optimize import BFGS


atoms = Atoms('Si2', positions=[[5., 5., 5.], [7., 5., 5.]],
              cell=[12, 12, 12], pbc=False)

atoms.calc = DFTBPlus(slako_dir=datafiles_directory,
                      hamiltonian=dict(dftb=dict(
                          scc=False,
                          polynomialrepulsive='setforall {yes}')))

dyn = BFGS(atoms, logfile='-')
dyn.run(fmax=0.1)

e = atoms.get_potential_energy()
assert abs(e - -64.830901) < 1., e
