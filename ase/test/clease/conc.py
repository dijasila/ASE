import numpy as np
from ase.clease.concentration import Concentration


def test_full_range():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    conc = Concentration(basis_elements=basis_elements)
    conc = conc.get_random_concentration()
    sum1 = np.sum(conc[:3])
    assert abs(sum1 - 1) < 1E-9
    sum2 = np.sum(conc[3:])
    assert abs(sum2 - 1) < 1E-9


def fixed_composition():
    basis_elements = [['Li', 'Ru'], ['O', 'X']]
    A_eq = [[0, 3, 0, 0], [0, 0, 0, 2]]
    b_eq = [1, 1]
    conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
    rand = conc.get_random_concentration()
    assert np.allclose(rand, np.array([2./3, 1./3, 0.5, 0.5]))


def test_conc_range():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    conc = Concentration(basis_elements=basis_elements)
    ranges = [[(0, 1), (1./3, 1./3), (0, 1)], [(2./3, 1), (0, 1)]]
    conc.set_conc_ranges(ranges)
    rand = conc.get_random_concentration()
    sum1 = np.sum(rand[:3])
    assert abs(sum1 - 1) < 1E-9
    sum2 = np.sum(rand[3:])
    assert abs(sum2 - 1) < 1E-9
    assert abs(rand[1] - 1./3) < 1E-9
    assert rand[3] > 2./3

def test_fix_Ru_composition():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    A_eq = [[0, 3, 0, 0, 0]]
    b_eq = [1]
    A_lb = [[0, 0, 0, 3, 0]]
    b_lb = [2]
    conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq,
                         A_lb=A_lb, b_lb=b_lb)
    conc = conc.get_random_concentration()
    sum1 = np.sum(conc[:3])
    assert abs(sum1 - 1) < 1E-9
    sum2 = np.sum(conc[3:])
    assert abs(sum2 - 1) < 1E-9


test_full_range()
fixed_composition()
test_conc_range()
test_fix_Ru_composition()
