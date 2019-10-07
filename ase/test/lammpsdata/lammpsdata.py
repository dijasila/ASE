#!/usr/bin/env python3

import ase.io
from ase.utils import StringIO

input = StringIO("""LAMMPS data file via write_data, version 29 Mar 2019, timestep = 1000000

8 atoms
1 atom types
6 bonds
1 bond types
4 angles
1 angle types

-5.1188800000000001e+01 5.1188800000000001e+01 xlo xhi
-5.1188800000000001e+01 5.1188800000000001e+01 ylo yhi
-5.1188800000000001e+01 5.1188800000000001e+01 zlo zhi

Masses

1 56

Atoms

1 1 1 0.0 -7.0654012878945753e+00 -4.7737244253442213e-01 -5.1102452666801824e+01 2 -1 6
2 1 1 0.0 -8.1237844371679362e+00 -1.3340695922796841e+00 4.7658302278206179e+01 2 -1 5
3 1 1 0.0 -1.2090525219882498e+01 -3.2315354021627760e+00 4.7363437099502839e+01 2 -1 5
4 1 1 0.0 -8.3272244953257601e+00 -4.8413162043515321e+00 4.5609055410298623e+01 2 -1 5
5 2 1 0.0 -5.3879618209198750e+00 4.9524635221072280e+01 3.0054862714858366e+01 6 -7 -2
6 2 1 0.0 -8.4950075933508273e+00 -4.9363297129348325e+01 3.2588925816534982e+01 6 -6 -2
7 2 1 0.0 -9.7544282093133940e+00 4.9869755980935565e+01 3.6362287886934432e+01 6 -7 -2
8 2 1 0.0 -5.5712437770663756e+00 4.7660225526197003e+01 3.8847235874270240e+01 6 -7 -2

Velocities

1 -1.2812627962466232e-02 -1.8102422526771818e-03 8.8697845357364469e-03
2 7.7087896348612683e-03 -5.6149199730983867e-04 1.3646724560472424e-02
3 -3.5128553734623657e-03 1.2368758037550581e-03 9.7460093657088121e-03
4 1.1626059392751346e-02 -1.1942908859710665e-05 8.7505240354339674e-03
5 1.0953500823880464e-02 -1.6710422557096375e-02 2.2322216388444985e-03
6 3.7515599452757294e-03 1.4091708517087744e-02 7.2963916249300454e-03
7 5.3953961772651359e-03 -8.2013715102925017e-03 2.0159609509813853e-02
8 7.5074008407567160e-03 5.9398495239242483e-03 7.3144909044607909e-03

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
4 1 5 6
5 1 6 7
6 1 7 8

Angles

1 1 1 2 3
2 1 2 3 4
3 1 5 6 7
4 1 6 7 8
""")

at = ase.io.read(input, format="lammps-data",units="metal")

expected_output = ['8', 'Lattice="102.3776 0.0 0.0 0.0 102.3776 0.0 0.0 0.0 102.3776" Properties=species:S:1:pos:R:3:bonds:S:1:masses:R:1:mol-id:I:1:Z:I:1:angles:S:1:momenta:R:3:type:I:1:id:I:1:travel:I:3 pbc="T T T"', 'H      -7.06540129      -0.47737244     -51.10245267 {1: [[1]]}      56.00000001        1        1 {}      -0.00730459      -0.00103203       0.00505674        1        1        2       -1        6', 'H      -8.12378444      -1.33406959      47.65830228 {1: [[2]]}      56.00000001        1        1 {1: [[0, 2]]}       0.00439485      -0.00032011       0.00778011        1        2        2       -1        5', 'H     -12.09052522      -3.23153540      47.36343710 {1: [[3]]}      56.00000001        1        1 {1: [[1, 3]]}      -0.00200271       0.00070515       0.00555628        1        3        2       -1        5', 'H      -8.32722450      -4.84131620      45.60905541 {}      56.00000001        1        1 {}       0.00662811      -0.00000681       0.00498875        1        4        2       -1        5', 'H      -5.38796182      49.52463522      30.05486271 {1: [[5]]}      56.00000001        2        1 {}       0.00624468      -0.00952675       0.00127261        1        5        6       -7       -2', 'H      -8.49500759     -49.36329713      32.58892582 {1: [[6]]}      56.00000001        2        1 {1: [[4, 6]]}       0.00213880       0.00803380       0.00415973        1        6        6       -6       -2', 'H      -9.75442821      49.86975598      36.36228789 {1: [[7]]}      56.00000001        2        1 {1: [[5, 7]]}       0.00307596      -0.00467567       0.01149316        1        7        6       -7       -2', 'H      -5.57124378      47.66022553      38.84723587 {}      56.00000001        2        1 {}       0.00428003       0.00338636       0.00417005        1        8        6       -7       -2', '']

buf = StringIO()
ase.io.write(buf, at, format="extxyz",
             columns=["symbols",
                      "positions",
                      "bonds",
                      "masses",
                      "mol-ids",
                      "numbers",
                      "angles",
                      "momenta",
                      "types",
                      "ids",
                      "travel"],
             write_info=False)

lines = [line.strip() for line in buf.getvalue().split("\n")]
print(lines)

assert lines == expected_output
