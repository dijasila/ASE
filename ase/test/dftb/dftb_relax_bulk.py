from ase.build import bulk
from ase.calculators.dftb_new import DFTBPlus
from ase.test.testsuite import datafiles_directory
from ase.optimize import QuasiNewton
from ase.constraints import ExpCellFilter

atoms = bulk('Si')
atoms.calc = DFTBPlus(slako_dir=datafiles_directory,
                      kpts=(3, 3, 3),
                      hamiltonian=dict(scc=True))

dyn = QuasiNewton(ExpCellFilter(atoms))
dyn.run(fmax=0.01)

e = atoms.get_potential_energy()
assert abs(e - -73.150819) < 1., e
