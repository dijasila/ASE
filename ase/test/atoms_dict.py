import json
import warnings
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


atoms = Atoms("N2", [(0, 0, 0), (0, 0, 1)], cell=np.eye(3), pbc=[1, 0, 1])


def test_json_dump(atoms=atoms):
    # convert to json string
    dict1 = atoms.todict()
    json.dumps(dict1)


def test_json_dump_load(atoms=atoms):
    # convert to json string
    dict1 = atoms.todict()
    rep = json.dumps(dict1)

    # read back from json string
    dict2 = json.loads(rep)

    # make sure the dictionary is actually the same
    assert dict1 == dict2

    # the Atoms created from the dict are equal to the starting point
    new_atoms1 = Atoms(**dict2)
    new_atoms2 = Atoms.fromdict(dict2)

    assert atoms == new_atoms1
    assert atoms == new_atoms2


def test_json_dump_warnings(atoms=atoms):
    c = FixAtoms(indices=[0, 1])
    atoms.set_constraint(c)
    atoms.info = {"arbitrary": "something"}

    with warnings.catch_warnings(record=True) as w:
        atoms.todict(warn=False)
        assert len(w) == 0

    with warnings.catch_warnings(record=True) as w:
        json.dumps(atoms.todict())
        assert len(w) == 1


if __name__ == "__main__":
    test_json_dump()
    test_json_dump_load()
    test_json_dump_warnings()
