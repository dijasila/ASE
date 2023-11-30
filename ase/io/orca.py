from io import StringIO
from ase.io import read
from ase.utils import reader, writer
from ase.units import Hartree, Bohr
from pathlib import Path
import re

import numpy as np

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


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write(f"! {params['orcasimpleinput']} \n")
    fd.write(f"{params['orcablocks']} \n")

    fd.write('*xyz')
    fd.write(" %d" % params['charge'])
    fd.write(" %d \n" % params['mult'])
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


def read_charge(text):
    re_charge = re.compile(r'Sum of atomic charges\s*:\s*([+-]?[0-9]*\.[0-9]*)')

    match = None
    for match in re_charge.finditer(text):
        pass
    if match is None:
        raise RuntimeError('No charge')
    charge = float(match.group(1))

    return charge


def read_energy(text):
    re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
    re_not_converged = re.compile(r"Wavefunction not fully converged")

    found_line = re_energy.finditer(text)
    energy = float('nan')
    for match in found_line:
        if not re_not_converged.search(match.group()):
            energy = float(match.group().split()[-1]) * Hartree
    if np.isnan(energy):
        raise RuntimeError('No energy')

    return energy


def read_center_of_mass(text):
    """ Scan through text for the center of mass """
    # Example:
    # 'The origin for moment calculation is the CENTER OF MASS  =
    # ( 0.002150, -0.296255  0.086315)'
    # Note the missing comma in the output
    re_com = re.compile(r'The origin for moment calculation is the '
                        r'CENTER OF MASS\s+=\s+\('
                        r'\s+(-?[0-9]+\.[0-9]+)'
                        r',?\s+(-?[0-9]+\.[0-9]+)'
                        r',?\s+(-?[0-9]+\.[0-9]+)'
                        r'\)')

    match = None
    for match in re_com.finditer(text):
        pass
    if match is None:
        # Nothing was found
        return None

    # Return the last match
    com = np.array([float(s) for s in match.groups()]) * Bohr
    return com


def read_dipole(text):
    """ Scan through text for the dipole moment in the COM
    frame of reference """
    # Example:
    # 'Total Dipole Moment    :      3.15321      -0.00269       0.03656'
    re_dipole = re.compile(r'Total Dipole Moment\s+:'
                           r'\s+(-?[0-9]+\.[0-9]+)'
                           r'\s+(-?[0-9]+\.[0-9]+)'
                           r'\s+(-?[0-9]+\.[0-9]+)')

    match = None
    for match in re_dipole.finditer(text):
        pass
    if match is None:
        # Nothing was found
        return None

    # Return the last match
    dipole = np.array([float(s) for s in match.groups()]) * Bohr
    return dipole


@reader
def read_orca_output(fd):
    """ From the ORCA output file: Read Energy and dipole moment
    in the frame of reference of the center of mass "
    """
    text = fd.read()

    energy = read_energy(text)
    charge = read_charge(text)
    position_COM = read_center_of_mass(text)
    dipole_COM = read_dipole(text)

    results = dict()
    results['energy'] = energy
    results['free_energy'] = energy

    if dipole_COM is not None:
        dipole = dipole_COM + position_COM * charge
        results['dipole'] = dipole

    return results


@reader
def read_orca_engrad(fd):
    """Read Forces from ORCA engrad file."""
    text = fd.read()
    re_gradient = re.compile(r'# The current gradient.*\n#\n')
    re_stop = re.compile(r'#\n# The at')
    re_values = re.compile(r'^\s*(-?[0-9]+\.[0-9]+)', re.MULTILINE)

    # Search for beginning of block
    match = re_gradient.search(text)
    if match is None:
        raise RuntimeError('No match for gradient')
    # Discard everything before this block
    text = text[match.end():]

    # Search for end of block
    match = re_stop.search(text)
    if match is None:
        raise RuntimeError('No match for atomic numbers and coordinates')
    # Discard everything after this block
    text = text[:match.start()]

    # Parse the values
    gradients = [float(match.group(0)) for match in re_values.finditer(text)]

    # Reshape
    gradients = np.array(gradients).reshape((-1, 3))

    results = dict()
    results['forces'] = -gradients * Hartree / Bohr
    return results


def read_orca_outputs(directory, stdout_path):
    stdout_path = Path(stdout_path)
    results = {}
    results.update(read_orca_output(stdout_path))

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = stdout_path.with_suffix('.engrad')
    if engrad_path.is_file():
        results.update(read_orca_engrad(engrad_path))
    return results
