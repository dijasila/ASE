from ase.test import cli, require
from ase.db import connect
from ase.io.jsonio import read_json
from ase.calculators.nwchem import NWChem

require('nwchem')


cli("""\
ase build O O.traj &&
ase run nwchem O.traj -o nwchem_cmdline.json &&
ase build O2 O2.traj &&
ase run nwchem O2.traj -o nwchem_cmdline.json""")
c = connect('nwchem_cmdline.json')
dct = read_json('nwchem_cmdline.json')
for name in ['O2', 'O']:
    d = c.get([('formula', '=', name)])
    id = d.id
    e1 = d.energy
    e2 = c.get_atoms(id).get_potential_energy()
    e3 = NWChem.read_atoms(name).get_potential_energy()
    e4 = dct[id]['energy']
    assert e1 == e2 == e3 == e4
    print(e1)
ae = 2 * c.get('formula=O').energy - c.get('formula=O2').energy
assert abs(ae - 6.6053) < 1e-4
