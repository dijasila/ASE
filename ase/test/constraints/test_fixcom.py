import pytest
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixCom
from ase.build import molecule


@pytest.mark.optimize
def test_fixcom():
    atoms = molecule('H2O')
    atoms.center(vacuum=4)
    atoms.calc = EMT()
    cold = atoms.get_center_of_mass()
    atoms.set_constraint(FixCom())

    opt = BFGS(atoms)
    opt.run(steps=5)

    cnew = atoms.get_center_of_mass()

    assert max(abs(cnew - cold)) < 1e-8
