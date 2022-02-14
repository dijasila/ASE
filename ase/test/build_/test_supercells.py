import pytest
import numpy as np
from ase.build import bulk
from ase.build.supercells import make_supercell


@pytest.fixture
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture(params=[
    bulk("NaCl", crystalstructure="rocksalt", a=4.0),
    bulk("NaCl", crystalstructure="rocksalt", a=4.0, cubic=True),
    bulk("Au", crystalstructure="fcc", a=4.0),
])
def prim(request):
    return request.param


@pytest.fixture(params=[
    3 * np.diag([1, 1, 1]),
    4 * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
    3 * np.diag([1, 2, 1]),
])
def P(request):
    return request.param


def test_make_supercell(prim, P):
    rep = int(round(np.linalg.det(P)))
    expected = rep * len(prim)
    sc = make_supercell(prim, P)
    assert len(sc) == expected
    assert list(sc.symbols) == list(prim.symbols) * rep


def test_make_supercells_arrays(prim, rng):
    P = 3 * np.diag([1, 1, 1])
    reps = int(round(np.linalg.det(P)))
    tags = list(range(len(prim)))
    momenta = rng.random((len(prim), 3))

    prim.set_tags(tags)
    prim.set_momenta(momenta)

    sc = make_supercell(prim, P)

    assert reps * len(prim) == len(sc.get_tags())
    assert all(sc.get_tags() == np.tile(tags, reps))
    assert np.allclose(sc[:len(prim)].get_momenta(), prim.get_momenta())
    assert np.allclose(sc.get_momenta(), np.tile(momenta, (reps, 1)))
