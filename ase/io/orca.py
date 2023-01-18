from io import StringIO
from ase.io import read
from ase.utils import reader

# Made from NWChem interface


@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif (line.startswith('end') and stopline == -1):
            stopline = index
        elif (line.startswith('*') and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    return atoms


def write_orca(atoms, **params):
    """Function to write ORCA input file
    """
    charge = params['charge']
    mult = params['mult']
    label = params['label']

    if 'pcpot' in params.keys():
        pcpot = params['pcpot']
        pcstring = '% pointcharges \"' +\
                   label + '.pc\"\n\n'
        params['orcablocks'] += pcstring
        pcpot.write_mmcharges(label)

    with open(label + '.inp', 'w') as fd:
        fd.write("! %s \n" % params['orcasimpleinput'])
        fd.write("%s \n" % params['orcablocks'])

        fd.write('*xyz')
        fd.write(" %d" % charge)
        fd.write(" %d \n" % mult)
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(symbol +
                     str(atom.position[0]) + ' ' +
                     str(atom.position[1]) + ' ' +
                     str(atom.position[2]) + '\n')
        fd.write('*\n')
