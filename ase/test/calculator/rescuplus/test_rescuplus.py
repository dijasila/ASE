"""Check that RESCU+ calculation can run."""
import pytest
from ase.build import bulk
from ase.calculators.rescuplus import Rescuplus

calc = pytest.mark.calculator

def verify(calc):
    assert calc.get_eigenvalues(spin=0, kpt=0) is not None
    assert calc.get_fermi_level() is not None
    assert calc.get_forces() is not None
    assert calc.get_ibz_k_points() is not None
    assert calc.get_k_point_weights() is not None
    assert calc.get_number_of_spins() is not None
    assert calc.get_stress() is not None


@calc('rescuplus')
def test_main(factory):
    atoms = bulk('Si')
    pp = [{"label": "Si", "path": "Si_AtomicData.mat"}]
    inp = {"system": {"cell": {"res": 0.25}, "kpoint": {"grid": [5, 5, 5]}}}
    inp["energy"] = {"frcReturn": True, "stressReturn": True}
    atoms.calc = factory.calc(input_data=inp, pseudopotentials=pp)
    atoms.get_potential_energy()
    verify(atoms.calc)
