import pytest
import numpy as np

from ase.build import molecule
from ase.constraints import FixAtoms
from ase.optimize.internal import Internal


@pytest.mark.calculator_lite
@pytest.mark.calculator('dftb')
def test_dftb(factory, testdir):
    atoms = molecule('H2O')
    atoms[0].position[0] -= 0.5
    initial_positions = atoms.get_positions().copy()

    fixed = 0
    atoms.set_constraint(FixAtoms([fixed]))

    atoms.calc = factory.calc(label='h2o/main')

    fmax = 0.7
    dyn = Internal(atoms)
    dyn.run(fmax=fmax, steps=1000)

    assert (np.abs(atoms.get_forces()) < fmax).all()
    assert atoms.calc.name == 'dftb'
    params = atoms.calc.parameters
    assert 'Hamiltonian_SlaterKosterFiles_Prefix' in params

    # fixed should be the sameq
    assert atoms[fixed].position == pytest.approx(initial_positions[fixed],
                                                  1e-6)
    # others should have moved
    assert atoms.get_positions() != pytest.approx(initial_positions,
                                                  1e-6)
