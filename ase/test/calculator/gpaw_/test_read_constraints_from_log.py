"""A simple test for reading in the constraints given the gpaw logfile
I do not have access to the ase-datafiles repo, so I am adding the logfile
into the same directory
"""
from ase.io import read
# from pathlib import Path
import numpy as np


def test_gpaw_constraints_from_log():
    # parent = str(Path(__file__).parent)
    # gpaw_logfile = parent + '/gpaw_output_for_constraint_reading.txt'
    gpaw_logfile = 'gpaw_output.txt'

    write_gpaw_out(gpaw_logfile)
    atoms = read(gpaw_logfile)

    assert len(atoms.constraints) == 16

    constraints = {i: '' for i in range(len(atoms))}
    # Read the labels that should be in the positions table from
    # atoms.constraints
    for const in atoms.constraints:
        const_label = [i for i in const.todict()['name']
                       if i.lstrip('Fix').isupper()][0]

        indices = []
        for key, value in const.__dict__.items():
            # Since the indices in the varying constraints are labeled
            # differently we have to search for all the labels
            if key in ['a', 'index', 'pairs', 'indices']:
                indices = np.unique(np.array(value).reshape(-1))

        for index in indices:
            constraints[index] += const_label

    # Read the positions table to compare whether the labels have been
    # set correctly.

    infile = open(gpaw_logfile)
    lines = infile.readlines()
    infile.close()

    i1 = 0
    for i2, line in enumerate(lines):
        if i1 > 0:
            if not len(line.split()):
                i3 = i2
                break
        if len(line.split()):
            if line.split()[0].rstrip(':') == 'Positions':
                i1 = i2 + 1

    # Check if labels in the table correspond to the indices of the contraints
    for n, line in enumerate(lines[i1:i3]):
        assert constraints[n] == line.split()[5]


def write_gpaw_out(gpaw_logfile):
    out = open(gpaw_logfile, 'w')
    out.write('''  ___ ___ ___ _ _ _
 |   |   |_  | | | |
 | | | | | . | | | |
 |__ |  _|___|_____|  21.6.1b1
 |___|_|

Date:   Thu Oct 14 11:15:03 2021
Arch:   x86_64
Pid:    42184
Python: 3.8.6
libxc:  4.3.4
units:  Angstrom and eV
cores: 1
OpenMP: False
OMP_NUM_THREADS: 1

Input parameters:
  convergence: {bands: occupied,
                density: inf,
                eigenstates: 4e+100,
                energy: 5000}
  h: 0.4
  kpts: [1 1 1]

System changes: positions, numbers, cell, pbc, initial_charges, initial_magmoms

Initialize ...

Cu-setup:
  name: Copper
  id: 0f0e166f2a1531348bd10bcfa07bef11
  Z: 29.0
  valence: 11
  core: 18
  charge: 0.0
  file: /home/modules/software/GPAW-setups/0.9.20000/Cu.LDA.gz
  compensation charges: gauss, rc=0.33, lmax=2
  cutoffs: 2.06(filt), 2.43(core),
  valence states:
                energy  radius
    4s(1.00)    -4.857   1.164
    4p(0.00)    -0.783   1.164
    3d(10.00)    -5.324   1.058
    *s          22.354   1.164
    *p          26.429   1.164
    *d          21.887   1.058

  Using partial waves for Cu as LCAO basis

Reference energy: -179804.388190

Spin-paired calculation

Convergence criteria:
 Maximum [total energy] change in last 3 cyles: 5000 eV / electron
 Maximum integral of absolute [dens]ity change: inf electrons / valence electron
 Maximum number of scf [iter]ations: 333
 (Square brackets indicate name in SCF output, whereas a 'c' in
 the SCF output indicates the quantity has converged.)

Symmetries present (total): 4

  ( 1  0  0)  ( 0  1  0)  ( 0 -1  0)  (-1  0  0)
  ( 0  1  0)  ( 1  0  0)  (-1  0  0)  ( 0 -1  0)
  ( 0  0  1)  ( 0  0  1)  ( 0  0 -1)  ( 0  0 -1)

1 k-point (Gamma)
1 k-point in the irreducible part of the Brillouin zone
       k-points in crystal coordinates                weights
   0:     0.00000000    0.00000000    0.00000000          1/1

Wave functions: Uniform real-space grid
  Kinetic energy operator: 12*3+1=37 point O(h^6) finite-difference Laplacian
  ScaLapack parameters: grid=1x1, blocksize=None
  Wavefunction extrapolation:
    Improved wavefunction reuse through dual PAW basis

Occupation numbers: Fermi-Dirac: width=0.1000 eV


Eigensolver
   Davidson(niter=2)

Densities:
  Coarse grid: 12*12*4 grid
  Fine grid: 24*24*8 grid
  Total Charge: 0.000000

Density mixing:
  Method: separate
  Backend: pulay
  Linear mixing parameter: 0.05
  Mixing with 5 old densities
  Damping of long wave oscillations: 50

Hamiltonian:
  XC and Coulomb potentials evaluated on a 24*24*8 grid
  Using the LDA Exchange-Correlation functional
  Interpolation: tri-quintic (5. degree polynomial)
  Poisson solver: FastPoissonSolver using
    Stencil: 12*3+1=37 point O(h^6) finite-difference Laplacian
    FFT axes: [0, 1, 2]
    FST axes: []


ASE contraints:
  FixAtoms(indices=[0, 2]) (A)
  FixBondLengths(pairs=[[1, 2], [0, 2]], tolerance=1e-13) (B)
  FixCom()
  FixedPlane(indices=[2], direction=[0.0, 0.0, 1.0]) (P)
  FixedLine(indices=[2], direction=[0.0, 0.0, 1.0]) (L)
  FixCartesian(a=[2], mask=[False, False, False]) (C)
  FixScaled([1], True) (S)
  FixInternals(bonds=[2, [0, 1]],angles_deg=[90, [0, 1, 2]],epsilon=1e-07)
  FixedMode([0.3713906763541037, 0.5570860145311556, 0.7427813527082074])
  FixLinearTriatomic(triples=[[0, 1, 2]])
  ExternalForce(1, 0, 1.000000) (E)
  Hookean(0, 3, k=3, rt=3) (H)
  Hookean(3, [0, 2, 1], k=3, rt=3) (H)
  Hookean(3, [0, 2, 1, 0.5], k=3, rt=3) (H)
  MirrorForce(1, 0, 2.500000, 1.000000, 0.100000) (M)
  MirrorTorque(0, 1, 2, 3, 6.283185, 0.000000, 0.100000) (M)

Memory estimate:
  Process memory now: 69.03 MiB
  Calculator: 2.05 MiB
    Density: 1.30 MiB
      Arrays: 0.11 MiB
      Localized functions: 1.15 MiB
      Mixer: 0.04 MiB
    Hamiltonian: 0.16 MiB
      Arrays: 0.07 MiB
      XC: 0.00 MiB
      Poisson: 0.00 MiB
      vbar: 0.09 MiB
    Wavefunctions: 0.58 MiB
      Arrays psit_nG: 0.14 MiB
      Eigensolver: 0.23 MiB
      Projections: 0.02 MiB
      Projectors: 0.19 MiB

Total number of cores used: 1

Number of atoms: 4
Number of atomic orbitals: 36
Number of bands in calculation: 31
Number of valence electrons: 44
Bands to converge: occupied

... initialized

Initializing position-dependent things.

Density initialized from atomic densities
Creating initial wave functions:
  31 bands from LCAO basis set





              Cu

         Cu

            Cu

       Cu





                       Positions             Constr           (Magmoms)
   0 Cu     0.000000    0.000000    0.000000 ABEHMM ( 0.0000,  0.0000,  0.0000)
   1 Cu     1.950000    0.000000    1.950000  BSEMM ( 0.0000,  0.0000,  0.0000)
   2 Cu     0.000000    1.950000    1.950000 ABPLCM ( 0.0000,  0.0000,  0.0000)
   3 Cu     1.950000    1.950000    3.900000  HHHM  ( 0.0000,  0.0000,  0.0000)

Unit cell:
           periodic     x           y           z      points  spacing
  1. axis:    yes    0.000000    3.900000    3.900000    12     0.3753
  2. axis:    yes    3.900000    0.000000    3.900000    12     0.3753
  3. axis:    yes    1.950000    1.950000    0.000000     4     0.5629

  Lengths:   5.515433   5.515433   2.757716
  Angles:   60.000000  60.000000  60.000000

Effective grid spacing dv^(1/3) = 0.4687

iter     time        total  log10-change:
                    energy  eigst   dens
   1 11:15:04     9.274520
   2 11:15:04     3.823180  +1.14c -0.74c
   3 11:15:04     6.175967c +0.77c -0.78c

Converged after 3 iterations.

Dipole moment: (-8.724879, -8.724879, -7.008761) |e|*Ang

Energy contributions relative to reference atoms: (reference = -179804.388190)

Kinetic:       -119.440939
Potential:     +112.455094
External:        +0.000000
XC:              +1.562331
Entropy (-ST):   -0.004995
Local:          +11.601979
--------------------------
Free energy:     +6.173469
Extrapolated:    +6.175967

 Band  Eigenvalues  Occupancy
    0     -3.20576    2.00000
    1     -1.31649    2.00000
    2     -1.31626    2.00000
    3     -1.12803    2.00000
    4     -0.63599    2.00000
    5      1.68349    2.00000
    6      1.68617    2.00000
    7      2.18944    2.00000
    8      2.63275    2.00000
    9      3.03913    2.00000
   10      3.53589    2.00000
   11      3.78711    2.00000
   12      3.79139    2.00000
   13      4.69285    2.00000
   14      4.69337    2.00000
   15      5.61353    2.00000
   16      5.61761    2.00000
   17      6.13291    2.00000
   18      6.65766    2.00000
   19      6.69060    2.00000
   20      8.17133    1.99840
   21      8.17183    1.99839
   22      9.52757    0.00321
   23     13.06283    0.00000
   24     13.06652    0.00000
   25     13.65611    0.00000
   26     16.43029    0.00000
   27     16.51356    0.00000
   28     20.17187    0.00000
   29     23.55345    0.00000
   30     23.55449    0.00000

Fermi level: 8.88427

Gap: 1.356 eV
Transition (v -> c):
  (s=0, k=0, n=21, [0.00, 0.00, 0.00]) -> (s=0, k=0, n=22, [0.00, 0.00, 0.00])
Timing:                              incl.     excl.
-----------------------------------------------------------
Hamiltonian:                         0.086     0.000   0.0% |
 Atomic:                             0.081     0.001   0.1% |
  XC Correction:                     0.080     0.080   7.6% |--|
 Calculate atomic Hamiltonians:      0.001     0.001   0.1% |
 Communicate:                        0.000     0.000   0.0% |
 Hartree integrate/restrict:         0.000     0.000   0.0% |
 Initialize Hamiltonian:             0.000     0.000   0.0% |
 Poisson:                            0.003     0.000   0.0% |
  Communicate from 1D:               0.001     0.001   0.1% |
  Communicate from 2D:               0.001     0.001   0.1% |
  Communicate to 1D:                 0.001     0.001   0.1% |
  Communicate to 2D:                 0.001     0.001   0.1% |
  FFT 1D:                            0.000     0.000   0.0% |
  FFT 2D:                            0.001     0.001   0.1% |
 XC 3D grid:                         0.001     0.001   0.1% |
 vbar:                               0.000     0.000   0.0% |
LCAO initialization:                 0.354     0.070   6.7% |--|
 LCAO eigensolver:                   0.039     0.000   0.0% |
  Calculate projections:             0.000     0.000   0.0% |
  DenseAtomicCorrection:             0.000     0.000   0.0% |
  Distribute overlap matrix:         0.000     0.000   0.0% |
  Orbital Layouts:                   0.001     0.001   0.1% |
  Potential matrix:                  0.038     0.038   3.6% ||
 LCAO to grid:                       0.019     0.019   1.8% ||
 Set positions (LCAO WFS):           0.226     0.042   4.0% |-|
  Basic WFS set positions:           0.001     0.001   0.1% |
  Basis functions set positions:     0.000     0.000   0.0% |
  P tci:                             0.055     0.055   5.3% |-|
  ST tci:                            0.068     0.068   6.5% |--|
  mktci:                             0.061     0.061   5.8% |-|
SCF-cycle:                           0.255     0.009   0.9% |
 Davidson:                           0.091     0.061   5.9% |-|
  Apply hamiltonian:                 0.003     0.003   0.3% |
  Subspace diag:                     0.007     0.000   0.0% |
   calc_h_matrix:                    0.005     0.001   0.1% |
    Apply hamiltonian:               0.003     0.003   0.3% |
   diagonalize:                      0.001     0.001   0.1% |
   rotate_psi:                       0.001     0.001   0.1% |
  calc. matrices:                    0.014     0.007   0.7% |
   Apply hamiltonian:                0.007     0.007   0.6% |
  diagonalize:                       0.005     0.005   0.5% |
  rotate_psi:                        0.001     0.001   0.1% |
 Density:                            0.005     0.000   0.0% |
  Atomic density matrices:           0.002     0.002   0.2% |
  Mix:                               0.003     0.003   0.2% |
  Multipole moments:                 0.000     0.000   0.0% |
  Pseudo density:                    0.001     0.000   0.0% |
   Symmetrize density:               0.000     0.000   0.0% |
 Hamiltonian:                        0.145     0.000   0.0% |
  Atomic:                            0.136     0.002   0.2% |
   XC Correction:                    0.134     0.134  12.8% |----|
  Calculate atomic Hamiltonians:     0.002     0.002   0.2% |
  Communicate:                       0.000     0.000   0.0% |
  Hartree integrate/restrict:        0.000     0.000   0.0% |
  Poisson:                           0.006     0.000   0.0% |
   Communicate from 1D:              0.001     0.001   0.1% |
   Communicate from 2D:              0.001     0.001   0.1% |
   Communicate to 1D:                0.001     0.001   0.1% |
   Communicate to 2D:                0.001     0.001   0.1% |
   FFT 1D:                           0.000     0.000   0.0% |
   FFT 2D:                           0.001     0.001   0.1% |
  XC 3D grid:                        0.002     0.002   0.1% |
  vbar:                              0.000     0.000   0.0% |
 Orthonormalize:                     0.004     0.000   0.0% |
  calc_s_matrix:                     0.000     0.000   0.0% |
  inverse-cholesky:                  0.000     0.000   0.0% |
  projections:                       0.004     0.004   0.4% |
  rotate_psi_s:                      0.000     0.000   0.0% |
Set symmetry:                        0.002     0.002   0.2% |
Other:                               0.346     0.346  33.2% |------------|
-----------------------------------------------------------
Total:                                         1.044 100.0%

Date: Thu Oct 14 11:15:04 2021
''')
    out.close()

# test_gpaw_constraints_from_log()
