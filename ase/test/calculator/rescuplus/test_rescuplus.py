"""Check that RESCU calculation can run."""

from ase.build import bulk
from ase.calculators.rescu import Rescuplus


def verify(calc):
    assert calc.get_eigenvalues(spin=0, kpt=0) is not None
    assert calc.get_fermi_level() is not None
    assert calc.get_forces() is not None
    assert calc.get_ibz_k_points() is not None
    assert calc.get_k_point_weights() is not None
    assert calc.get_number_of_spins() is not None
    assert calc.get_stress() is not None


def test_main():
    atoms = bulk('Si')
    pp = [{"label": "Si", "path": "Si_AtomicData.mat"}]
    input_data = {"system": {"cell": {"res": 0.25}, "kpoint": {"grid": [5, 5, 5]}}}
    input_data["energy"] = {"frcReturn": True, "stressReturn": True}
    rsccmd = "mpiexec -n 1 rescuplus_scf -i PREFIX.rsi > resculog.out && cp rescu_scf_out.json PREFIX.rso"
    atoms.calc = Rescuplus(command=rsccmd, input_data=input_data, pseudopotentials=pp)
    atoms.get_potential_energy()
    verify(atoms.calc)
