import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import compare_atoms
from ase.io.orca import write_orca, read_geom_orcainp, read_orca_outputs


def test_orca_inputfile():
    sample_inputfile = """ ! engrad B3LYP def2-TZVPP
 %pal nprocs 4 end
 *xyz 0 1
 O   0.0 0.0 0.0
 H   1.0 0.0 0.0
 H   0.0 1.0 0.0
 *
"""
    sample_inputfile_lines = sample_inputfile.splitlines()

    atoms = Atoms('OHH', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    kw = dict(charge=0, mult=1,
              orcasimpleinput='engrad B3LYP def2-TZVPP',
              orcablocks='%pal nprocs 4 end')
    write_orca('orca.inp', atoms, kw)

    with open('orca.inp') as fd:
        lines = fd.readlines()

    assert len(lines) == len(sample_inputfile_lines)
    for line, expected_line in zip(lines, sample_inputfile_lines):
        assert line.strip() == expected_line.strip()


def test_read_geom_orcainp():
    atoms = Atoms('OHH', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    kw = dict(charge=0, mult=1,
              orcasimpleinput='B3LYP def2-TZVPP',
              orcablocks='%pal nprocs 4 end')

    fname = 'orcamolecule_test.inp'
    write_orca(fname, atoms, kw)

    with open(fname) as test:
        atoms2 = read_geom_orcainp(test)

    assert not compare_atoms(atoms, atoms2, tol=1e-7)


def test_read_orca_outputs():
    sample_outputfile = """\
--------------------------------------------
MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
--------------------------------------------
   0 F :   -0.223492    0.725756
   1 Si:    0.515710   -0.002768
   2 C :   -0.452154    0.089331
   3 C :   -0.696844    0.022858
   4 C :   -0.694442    0.017446
   5 C :   -0.698397    0.000030
   6 H :    0.207679   -0.001384
   7 H :    0.184093   -0.000681
   8 H :    0.184873   -0.000435
   9 H :    0.184661   -0.000913
  10 H :    0.212080   -0.002038
  11 H :    0.185083   -0.000575
  12 H :    0.182349    0.000294
  13 H :    0.181951    0.000082
  14 H :    0.209136   -0.000591
  15 C :    0.381353    0.032364
  16 C :   -0.006580    0.031356
  17 C :   -0.123313    0.027413
  18 C :   -0.141712   -0.010118
  19 C :   -0.117620    0.060307
  20 C :   -0.141521   -0.009931
  21 C :   -0.123495    0.027134
  22 H :    0.162169   -0.001285
  23 H :    0.155246    0.000176
  24 H :    0.156288   -0.002719
  25 H :    0.155160    0.000168
  26 H :    0.161737   -0.001275
Sum of atomic charges         :   -0.0000000
Sum of atomic spin populations:    1.0000000

 -------
TIMINGS
-------

Total SCF time: 0 days 0 hours 0 min 3 sec

Total time                  ....       3.805 sec
Sum of individual times     ....       3.040 sec  ( 79.9%)

Fock matrix formation       ....       2.425 sec  ( 63.7%)
  XC integration            ....       0.095 sec  (  3.9% of F)
    Basis function eval.    ....       0.020 sec  ( 21.5% of XC)
    Density eval.           ....       0.013 sec  ( 13.9% of XC)
    XC-Functional eval.     ....       0.012 sec  ( 12.7% of XC)
    XC-Potential eval.      ....       0.022 sec  ( 23.1% of XC)
Diagonalization             ....       0.004 sec  (  0.1%)
Density matrix formation    ....       0.047 sec  (  1.2%)
Population analysis         ....       0.002 sec  (  0.1%)
Initial guess               ....       0.470 sec  ( 12.3%)
Orbital Transformation      ....       0.000 sec  (  0.0%)
Orbital Orthonormalization  ....       0.000 sec  (  0.0%)
DIIS solution               ....       0.001 sec  (  0.0%)
SOSCF solution              ....       0.010 sec  (  0.3%)
Grid generation             ....       0.081 sec  (  2.1%)

-------------------------   --------------------
FINAL SINGLE POINT ENERGY       -76.422436201230
-------------------------   --------------------
"""

    sample_engradfile = """\
#
# Number of atoms
#
 3
#
# The current total energy in Eh
#
    -76.422436201230
#
# The current gradient in Eh/bohr
#
      -0.047131484960
      -0.047131484716
       0.000000000053
       0.025621056182
       0.021510428527
       0.000000000034
       0.021510428778
       0.025621056189
      -0.000000000087
#
# The atomic numbers and current coordinates in Bohr
#
   8     0.0000000    0.0000000    0.0000000
   1     1.8897261    0.0000000    0.0000000
   1     0.0000000    1.8897261    0.0000000
"""
    with open('orcamolecule_test.out', 'w') as fd:
        fd.write(sample_outputfile)

    with open('orcamolecule_test.engrad', 'w') as engrad:
        engrad.write(sample_engradfile)

    results_sample = {
        'energy': -2079.560412394247,
        'forces': np.array([
            [2.42359838e+00, 2.42359837e+00, -2.72536956e-09],
            [-1.31748767e+00, -1.10611070e+00, -1.74835028e-09],
            [-1.10611071e+00, -1.31748767e+00, 4.47371984e-09]])}

    results_sample['free_energy'] = results_sample['energy']

    results = read_orca_outputs('.', 'orcamolecule_test.out')

    keys = set(results)
    assert keys == set(results_sample)

    for key in keys:
        # each result can be either float or ndarray.
        assert results[key] == pytest.approx(results_sample[key])


def test_read_orca_outputs_with_dipole(datadir):
    results_sample = {
        'energy': -22203.395391088576,
        'dipole': np.array([1.66861746, -0.00142348670, 0.0193467188])}

    results_sample['free_energy'] = results_sample['energy']

    results = read_orca_outputs('.', datadir / 'orca/orca_out_tail.out')

    keys = set(results)
    assert keys == set(results_sample)

    for key in keys:
        # each result can be either float or ndarray.
        assert results[key] == pytest.approx(results_sample[key])
