from ase.build import bulk
from ase.calculators.dftb_new import DFTBPlus
from ase.dft.dos import DOS
from ase.test.testsuite import datafiles_directory

atoms = bulk('Si')

atoms.calc = DFTBPlus(slako_dir=datafiles_directory,
                      kpts=(3, 3, 3),
                      hamiltonian=dict(scc=True,
                                       scctolerance=1e-5,
                                       maxangularmomentum=dict(si='d')))

atoms.get_potential_energy()

# efermi belongs to DFTBPlus.calc, which is a SinglePointDFTCalculator
efermi = atoms.calc.calc.get_fermi_level()
assert abs(efermi - -2.90086680996455) < 1.

# Similar to efermi, the necessary data belongs to DFTBPlus.calc
dos = DOS(atoms.calc.calc, width=0.2)
d = dos.get_dos()
e = dos.get_energies()
# DOS doesn't have a plot method?

atoms.calc = DFTBPlus(slako_dir=datafiles_directory,
                      kpts={'path': 'WGXWLG', 'npoints': 50},
                      hamiltonian=dict(scc=True,
                                       maxscciterations=1,
                                       readinitialcharges=True,
                                       maxangularmomentum=dict(si='d')))

atoms.calc.calculate(atoms)

bs = atoms.calc.band_structure()
