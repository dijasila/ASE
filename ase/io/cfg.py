import numpy as np

import ase
from ase.data import chemical_symbols
from ase.utils import reader, writer

cfg_default_fields = np.array(['positions', 'momenta', 'numbers', 'magmoms'])


@writer
def write_cfg(fd, atoms):
    """Write atomic configuration to a CFG-file (native AtomEye format).
    See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """

    fd.write('Number of particles = %i\n' % len(atoms))
    fd.write('A = 1.0 Angstrom\n')
    cell = atoms.get_cell(complete=True)
    for i in range(3):
        for j in range(3):
            fd.write('H0(%1.1i,%1.1i) = %f A\n' % (i + 1, j + 1, cell[i, j]))

    entry_count = 3
    for x in atoms.arrays.keys():
        if x not in cfg_default_fields:
            if len(atoms.get_array(x).shape) == 1:
                entry_count += 1
            else:
                entry_count += atoms.get_array(x).shape[1]

    vels = atoms.get_velocities()
    if isinstance(vels, np.ndarray):
        entry_count += 3
    else:
        fd.write('.NO_VELOCITY.\n')

    fd.write('entry_count = %i\n' % entry_count)

    i = 0
    for name, aux in atoms.arrays.items():
        if name not in cfg_default_fields:
            if len(aux.shape) == 1:
                fd.write('auxiliary[%i] = %s [a.u.]\n' % (i, name))
                i += 1
            else:
                if aux.shape[1] == 3:
                    for j in range(3):
                        fd.write(
                            'auxiliary[%i] = %s_%s [a.u.]\n'
                            % (i, name, chr(ord('x') + j))
                        )
                        i += 1

                else:
                    for j in range(aux.shape[1]):
                        fd.write(
                            'auxiliary[%i] = %s_%1.1i [a.u.]\n' % (i, name, j)
                        )
                        i += 1

    # Distinct elements
    spos = atoms.get_scaled_positions()
    for i in atoms:
        el = i.symbol

        fd.write('%f\n' % ase.data.atomic_masses[chemical_symbols.index(el)])
        fd.write(f'{el}\n')

        x, y, z = spos[i.index, :]
        s = f'{x:e} {y:e} {z:e} '

        if isinstance(vels, np.ndarray):
            vx, vy, vz = vels[i.index, :]
            s = s + f' {vx:e} {vy:e} {vz:e} '

        for name, aux in atoms.arrays.items():
            if name not in cfg_default_fields:
                if len(aux.shape) == 1:
                    s += ' %e' % aux[i.index]
                else:
                    s += (aux.shape[1] * ' %e') % tuple(aux[i.index].tolist())

        fd.write(f'{s}\n')


default_color = {
    'H': [0.800, 0.800, 0.800],
    'C': [0.350, 0.350, 0.350],
    'O': [0.800, 0.200, 0.200],
}

default_radius = {'H': 0.435, 'C': 0.655, 'O': 0.730}


def write_clr(fd, atoms):
    """Write extra color and radius code to a CLR-file (for use with AtomEye).
    Hit F12 in AtomEye to use.
    See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    color = None
    radius = None
    if atoms.has('color'):
        color = atoms.get_array('color')
    if atoms.has('radius'):
        radius = atoms.get_array('radius')

    if color is None:
        color = np.zeros([len(atoms), 3], dtype=float)
        for a in atoms:
            color[a.index, :] = default_color[a.symbol]

    if radius is None:
        radius = np.zeros(len(atoms), dtype=float)
        for a in atoms:
            radius[a.index] = default_radius[a.symbol]

    radius.shape = (-1, 1)

    for c1, c2, c3, r in np.append(color, radius, axis=1):
        fd.write(f'{c1:f} {c2:f} {c3:f} {r:f}\n')


@reader
def read_cfg(fd):
    """Read atomic configuration from a CFG-file (native AtomEye format).
    See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    nat = None
    naux = 0
    aux = None
    auxstrs = None

    cell = np.zeros([3, 3])
    transform = np.eye(3)
    eta = np.zeros([3, 3])

    current_atom = 0
    current_symbol = None
    current_mass = None

    L = fd.readline()
    while L:
        L = L.strip()
        if len(L) != 0 and not L.startswith('#'):
            if L == '.NO_VELOCITY.':
                vels = None
                naux += 3
            else:
                s = L.split('=')
                if len(s) == 2:
                    key, value = s
                    key = key.strip()
                    value = [x.strip() for x in value.split()]
                    if key == 'Number of particles':
                        nat = int(value[0])
                        spos = np.zeros([nat, 3])
                        masses = np.zeros(nat)
                        syms = [''] * nat
                        vels = np.zeros([nat, 3])
                        if naux > 0:
                            aux = np.zeros([nat, naux])
                    elif key == 'A':
                        pass  # unit = float(value[0])
                    elif key == 'entry_count':
                        naux += int(value[0]) - 6
                        auxstrs = [''] * naux
                        if nat is not None:
                            aux = np.zeros([nat, naux])
                    elif key.startswith('H0('):
                        i, j = (int(x) for x in key[3:-1].split(','))
                        cell[i - 1, j - 1] = float(value[0])
                    elif key.startswith('Transform('):
                        i, j = (int(x) for x in key[10:-1].split(','))
                        transform[i - 1, j - 1] = float(value[0])
                    elif key.startswith('eta('):
                        i, j = (int(x) for x in key[4:-1].split(','))
                        eta[i - 1, j - 1] = float(value[0])
                    elif key.startswith('auxiliary['):
                        i = int(key[10:-1])
                        auxstrs[i] = value[0]
                else:
                    # Everything else must be particle data.
                    # First check if current line contains an element mass or
                    # name. Then we have an extended XYZ format.
                    s = [x.strip() for x in L.split()]
                    if len(s) == 1:
                        if L in chemical_symbols:
                            current_symbol = L
                        else:
                            current_mass = float(L)
                    elif current_symbol is None and current_mass is None:
                        # Standard CFG format
                        masses[current_atom] = float(s[0])
                        syms[current_atom] = s[1]
                        spos[current_atom, :] = [float(x) for x in s[2:5]]
                        vels[current_atom, :] = [float(x) for x in s[5:8]]
                        current_atom += 1
                    elif (
                        current_symbol is not None and current_mass is not None
                    ):
                        # Extended CFG format
                        masses[current_atom] = current_mass
                        syms[current_atom] = current_symbol
                        props = [float(x) for x in s]
                        spos[current_atom, :] = props[0:3]
                        off = 3
                        if vels is not None:
                            off = 6
                            vels[current_atom, :] = props[3:6]
                        aux[current_atom, :] = props[off:]
                        current_atom += 1
        L = fd.readline()

    # Sanity check
    if current_atom != nat:
        raise RuntimeError(
            'Number of atoms reported for CFG file (={}) and '
            'number of atoms actually read (={}) differ.'.format(
                nat, current_atom
            )
        )

    if np.any(eta != 0):
        raise NotImplementedError(
            'eta != 0 not yet implemented for CFG ' 'reader.'
        )
    cell = np.dot(cell, transform)

    if vels is None:
        a = ase.Atoms(
            symbols=syms,
            masses=masses,
            scaled_positions=spos,
            cell=cell,
            pbc=True,
        )
    else:
        a = ase.Atoms(
            symbols=syms,
            masses=masses,
            scaled_positions=spos,
            momenta=masses.reshape(-1, 1) * vels,
            cell=cell,
            pbc=True,
        )

    i = 0
    while i < naux:
        auxstr = auxstrs[i]
        if auxstr[-2:] == '_x':
            a.set_array(auxstr[:-2], aux[:, i : i + 3])
            i += 3
        else:
            a.set_array(auxstr, aux[:, i])
            i += 1

    return a
