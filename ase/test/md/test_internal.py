import pytest
import numpy as np

from ase.units import fs, kB
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.md.internal import Internal
from ase.io import Trajectory


@pytest.mark.calculator_lite
@pytest.mark.calculator('dftb')
def test_dftb(factory, testdir):
    atoms = molecule('H2O')
    initial_positions = atoms.get_positions().copy()

    fixed = 0
    atoms.set_constraint(FixAtoms([fixed]))

    atoms.calc = factory.calc(label='h2o/main')

    timestep = 2 * fs
    trajname = 'h2o.traj'
    dyn = Internal(atoms, timestep, 300 * kB, trajectory=trajname)

    steps0 = 5
    dyn.run(steps0)

    with Trajectory('h2o.traj') as traj:
        assert len(traj) == steps0

    # second run should concatenate
    steps1 = 3
    dyn.run(steps1)

    with Trajectory('h2o.traj') as traj:
        assert len(traj) == steps0 + steps1

        for atoms in traj:
            assert atoms.calc.name == 'dftb'
            assert atoms[fixed].position == pytest.approx(
                initial_positions[fixed], 1e-6)
            assert np.linalg.norm(atoms.get_dipole_moment()) != 0
