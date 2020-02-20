import os
import warnings
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from ase.atoms import Atoms
from ase.units import Hartree, Bohr, Debye
from ase.utils import reader
from ase.calculators.calculator import kpts2sizeandoffsets, kpts2ndarray
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                         SinglePointKPoint)


def read_dftb(filename='dftb_in.hsd'):
    """Method to read coordinates form DFTB+ input file dftb_in.hsd
    additionally read information about fixed atoms
    and periodic boundary condition
    """
    with open(filename, 'r') as myfile:
        lines = myfile.readlines()

    atoms_pos = []
    atom_symbols = []
    type_names = []
    my_pbc = False
    fractional = False
    mycell = []

    for iline, line in enumerate(lines):
        if (line.strip().startswith('#')):
            pass
        elif ('genformat') in line.lower():
            natoms = int(lines[iline + 1].split()[0])
            if lines[iline + 1].split()[1].lower() == 's':
                my_pbc = True
            elif lines[iline + 1].split()[1].lower() == 'f':
                my_pbc = True
                fractional = True
            symbols = lines[iline + 2].split()
            for i in range(natoms):
                index = iline + 3 + i
                aindex = int(lines[index].split()[1]) - 1
                atom_symbols.append(symbols[aindex])

                position = [float(p) for p in lines[index].split()[2:]]
                atoms_pos.append(position)
            if my_pbc:
                for i in range(3):
                    index = iline + 4 + natoms + i
                    cell = [float(c) for c in lines[index].split()]
                    mycell.append(cell)
        else:
            if ('TypeNames' in line):
                col = line.split()
                for i in range(3, len(col) - 1):
                    type_names.append(col[i].strip("\""))
            elif ('Periodic' in line):
                if ('Yes' in line):
                    my_pbc = True
            elif ('LatticeVectors' in line):
                for imycell in range(3):
                    extraline = lines[iline + imycell + 1]
                    cols = extraline.split()
                    mycell.append(
                        [float(cols[0]), float(cols[1]), float(cols[2])])
            else:
                pass

    if not my_pbc:
        mycell = [1.0, 1.0, 1.0]

    start_reading_coords = False
    stop_reading_coords = False
    for line in lines:
        if (line.strip().startswith('#')):
            pass
        else:
            if ('TypesAndCoordinates' in line):
                start_reading_coords = True
            if start_reading_coords:
                if ('}' in line):
                    stop_reading_coords = True
            if (start_reading_coords and not (stop_reading_coords)
               and 'TypesAndCoordinates' not in line):
                typeindexstr, xxx, yyy, zzz = line.split()[:4]
                typeindex = int(typeindexstr)
                symbol = type_names[typeindex-1]
                atom_symbols.append(symbol)
                atoms_pos.append([float(xxx), float(yyy), float(zzz)])

    if fractional:
        atoms = Atoms(scaled_positions=atoms_pos, symbols=atom_symbols,
                      cell=mycell, pbc=my_pbc)
    elif not fractional:
        atoms = Atoms(positions=atoms_pos, symbols=atom_symbols,
                      cell=mycell, pbc=my_pbc)

    return atoms


def read_dftb_velocities(atoms, filename='geo_end.xyz'):
    """Method to read velocities (AA/ps) from DFTB+ output file geo_end.xyz
    """
    from ase.units import second
    # AA/ps -> ase units
    AngdivPs2ASE = 1.0/(1e-12*second)

    myfile = open(filename)

    lines = myfile.readlines()
    # remove empty lines
    lines_ok = []
    for line in lines:
        if line.rstrip():
            lines_ok.append(line)

    velocities = []
    natoms = len(atoms)
    last_lines = lines_ok[-natoms:]
    for iline, line in enumerate(last_lines):
        inp = line.split()
        velocities.append([float(inp[5])*AngdivPs2ASE,
                           float(inp[6])*AngdivPs2ASE,
                           float(inp[7])*AngdivPs2ASE])

    atoms.set_velocities(velocities)
    return atoms


def read_dftb_lattice(fileobj='md.out',images=None):
    """
    Read lattice vectors from MD and return them as a list. If a molecules are parsed add them there.
    """
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    if images is not None:
        append = True
        if hasattr(images, 'get_positions'):
            images = [images]
    else:
        append = False

    fileobj.seek(0)
    lattices = []
    for line in fileobj:
        if 'Lattice vectors' in line:
            vec = []
            for i in range(3): #DFTB+ only supports 3D PBC
                line = fileobj.readline().split()
                try:
                    line = [float(x) for x in line]
                except ValueError:
                    raise ValueError('Lattice vector elements should be of type float.')
                vec.extend(line)
            lattices.append(np.array(vec).reshape((3,3)))

    if append:
        if len(images) != len(lattices):
            raise ValueError('Length of images given does not match number of cell vectors found')

        for i,atoms in enumerate(images):
            atoms.set_cell(lattices[i])
            #DFTB+ only supports 3D PBC
            atoms.set_pbc(True)
        return
    else:
        return lattices




def write_dftb_velocities(atoms, filename='velocities.txt'):
    """Method to write velocities (in atomic units) from ASE
       to a file to be read by dftb+
    """
    from ase.units import AUT, Bohr
    # ase units -> atomic units
    ASE2au = Bohr / AUT

    if isinstance(filename, str):
        myfile = open(filename, 'w')
    else:
        # Assume it's a 'file-like object'
        myfile = filename

    velocities = atoms.get_velocities()
    for velocity in velocities:
        myfile.write(' %19.16f %19.16f %19.16f \n'
                     % (velocity[0] / ASE2au,
                        velocity[1] / ASE2au,
                        velocity[2] / ASE2au))

    return


def write_dftb(filename, atoms):
    """Method to write atom structure in DFTB+ format
       (gen format)
    """

    # sort
    atoms.set_masses()
    masses = atoms.get_masses()
    indexes = np.argsort(masses)
    atomsnew = Atoms()
    for i in indexes:
        atomsnew = atomsnew + atoms[i]

    if isinstance(filename, str):
        myfile = open(filename, 'w')
    else:
        # Assume it's a 'file-like object'
        myfile = filename

    ispbc = atoms.get_pbc()
    box = atoms.get_cell()

    if (any(ispbc)):
        myfile.write('%8d %2s \n' % (len(atoms), 'S'))
    else:
        myfile.write('%8d %2s \n' % (len(atoms), 'C'))

    chemsym = atomsnew.get_chemical_symbols()
    allchem = ''
    for i in chemsym:
        if i not in allchem:
            allchem = allchem + i + ' '
    myfile.write(allchem+' \n')

    coords = atomsnew.get_positions()
    itype = 1
    for iatom, coord in enumerate(coords):
        if iatom > 0:
            if chemsym[iatom] != chemsym[iatom-1]:
                itype = itype+1
        myfile.write('%5i%5i  %19.16f %19.16f %19.16f \n'
                     % (iatom+1, itype,
                        coords[iatom][0], coords[iatom][1], coords[iatom][2]))
    # write box
    if (any(ispbc)):
        # dftb dummy
        myfile.write(' %19.16f %19.16f %19.16f \n' % (0, 0, 0))
        myfile.write(' %19.16f %19.16f %19.16f \n'
                     % (box[0][0], box[0][1], box[0][2]))
        myfile.write(' %19.16f %19.16f %19.16f \n'
                     % (box[1][0], box[1][1], box[1][2]))
        myfile.write(' %19.16f %19.16f %19.16f \n'
                     % (box[2][0], box[2][1], box[2][2]))

    if isinstance(filename, str):
        myfile.close()


def _format_geom(atoms):
    symbol2num = OrderedDict()
    for i, symbol in enumerate(set(atoms.symbols)):
        symbol2num[symbol] = i + 1

    # out of principle, use lowercase everything. DFTB+ is case insensitive.
    out = ['geometry = {',
           '  typenames = { "' + '" "'.join(symbol2num) + '" }',
           '  typesandcoordinates [angstrom] = {']

    aline = '    {:>3} {:>24.16e} {:>24.16e} {:>24.16e}'
    for atom in atoms:
        out.append(aline.format(symbol2num[atom.symbol], *atom.position))

    out.append('  }')

    pbc = any(atoms.pbc)
    if not pbc:
        out.append('}')
        return '\n'.join(out)

    if not all(atoms.pbc):
        warnings.warn("DFTB+ only support aperiodic and fully 3D-periodic "
                      "simulations, but your system is periodic in only "
                      "one or two directions. This system will be treated "
                      "as 3D periodic instead!")

    out += ['  periodic = yes',
            '  latticevectors [angstrom] = {']

    vline = '    {:>24.16e} {:>24.16e} {:>24.16e}'
    for vec in atoms.cell:
        out.append(vline.format(*vec))

    out += ['  }',
            '}']
    return '\n'.join(out)


def _format_argument(key, val, nindent=0):
    head = '  ' * nindent
    secname = ''
    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], dict):
        secname = val[0] + ' '
        val = val[1]

    if isinstance(val, dict):
        out = ['{head}{key} = {secname}{{'.format(head=head, key=key.lower(),
                                                  secname=secname)]
        if not val:
            # Put the closing bracket on the same line if val is empty
            out.append('}')
            return ''.join(out)

        for subkey, subval in val.items():
            out.append(_format_argument(subkey, subval, nindent + 1))
        out.append('{head}}}'.format(head=head))
        return '\n'.join(out)

    if isinstance(val, bool):
        val = 'YES' if val else 'NO'
    elif isinstance(val, float):
        val = '{:>24.16e}'.format(val)
    elif isinstance(val, np.ndarray):
        if len(val.shape) > 1:
            raise ValueError("Don't know how to format array with shape {}!"
                             .format(val.shape))
        if val.dtype == bool:
            val = ' '.join(['YES' if x else 'NO' for x in val])
        elif val.dtype == float:
            # dftb+ input files are limited to 1024 chars per line
            # make sure everything fits
            width = min(24, 1024 // len(val))
            spec = '{{:>{}.{}e}}'.format(width, width - 8)
            val = ' '.join([spec.format(x) for x in val])
        else:
            val = ' '.join(map(str, val))

    return '{head}{key} = {val}'.format(head=head, key=key, val=val)


def _lowercase_dict(dict_in):
    dict_out = dict()
    for key, val in dict_in.items():
        if key.lower() in dict_out:
            raise ValueError("Section name {} has been provided more than "
                             "once!".format(key))
        key = key.lower()
        if key == 'kpts':
            dict_out[key] = deepcopy(val)
        elif isinstance(val, dict):
            dict_out[key] = _lowercase_dict(val)
        elif isinstance(val, tuple) and isinstance(val[1], dict):
            dict_out[key] = (val[0].lower(), _lowercase_dict(val[1]))
        elif isinstance(val, str):
            dict_out[key] = val.lower()
        else:
            dict_out[key] = deepcopy(val)
    return dict_out


def _format_kpts_mp(mesh, offset=(0., 0., 0.)):
    out = ['supercellfolding {']
    for row in np.diag(mesh):
        out.append('    {} {} {}'.format(*row))

    shifts = []
    shifts = np.array(mesh) * np.array(offset, dtype=float)
    shifts += (np.array(mesh) % 2 == 0) * 0.5
    out += ['    {} {} {}'.format(*shifts),
            '  }']
    return '\n'.join(out)


def _format_kpath(kpts):
    out = ['klines {']
    for kpt in kpts:
        out.append('    1 {:>24.16e} {:>24.16e} {:>24.16e}'.format(*kpt))
    out.append('  }')
    return '\n'.join(out)


def _format_kpts_literal(kpts):
    out = ['{']
    for kpt in kpts:
        out.append('    {:>24.16e} {:>24.16e} {:>24.16e} 1.0'.format(*kpt))
    out.append('  }')
    return '\n'.join(out)


def _format_kpts(atoms, kpts):
    if np.isscalar(kpts):
        # assume kpts is a density
        mesh, offset = kpts2sizeandoffsets(density=kpts, atoms=atoms)
        return _format_kpts_mp(mesh, offset)

    if isinstance(kpts, dict):
        if 'path' in kpts:
            return _format_kpath(kpts2ndarray(kpts, atoms=atoms))
        mesh, offset = kpts2sizeandoffsets(atoms=atoms, **kpts)
        return _format_kpts_mp(mesh, offset)

    if len(np.shape(kpts)) == 2:
        # assume a literal k-point specification
        return _format_kpts_literal(kpts)

    # otherwise, assume a MP mesh (e.g. (3, 3, 3))
    mesh, offset = kpts2sizeandoffsets(size=kpts, atoms=atoms)
    return _format_kpts_mp(mesh, offset)


def _merge_default_params(default, params):
    for key, val in default.items():
        if key not in params:
            params[key] = deepcopy(val)
        elif isinstance(val, dict) and isinstance(params[key], dict):
            _merge_default_params(val, params[key])
        elif isinstance(val, tuple):
            if isinstance(params[key], dict):
                _merge_default_params(val[1], params[key])
            elif isinstance(params[key], tuple) and val[0] == params[key][0]:
                _merge_default_params(val[0], params[key][0])


def write_dftb_in(fd, atoms, properties=None, default_params=None,
                  **params_in):
    out = [_format_geom(atoms)]

    if properties is None:
        properties = ['energy']

    # DFTB+ docs use CamelCase for settings, but the input file is actually
    # case insensitive. For consistency, use lowercase for everything.
    params = _lowercase_dict(params_in)
    _merge_default_params(default_params, params)

    ham = params.pop('hamiltonian', dict())
    kpts = params.pop('kpts', None)
    if kpts is not None:
        if 'kpointsandweights' in ham:
            warnings.warn("KPointsAndWeights found in settings, 'kpts' "
                          "argument will be ignored!")
        else:
            ham['kpointsandweights'] = _format_kpts(atoms, kpts)

    slako_dir = params.pop('slako_dir', None)
    skf = ham['slaterkosterfiles'][1]
    if slako_dir is not None:
        if 'prefix' in skf:
            warnings.warn("SlaterKosterFiles Prefix found in settings, "
                          "'slako_dir' argument will be ignored!")
        else:
            if slako_dir[-1] != '/':
                slako_dir += '/'
            skf['prefix'] = slako_dir

    skf_prefix = skf['prefix']
    skf_sep = skf['separator']
    skf_suffix = skf['suffix']

    # check if MaxAngularMomenta have been supplied manually; if not,
    # figure them out ourselves
    mam = ham['maxangularmomentum']
    for symbol in set(atoms.symbols):
        if symbol.lower() not in mam:
            mam[symbol.lower()] = _read_max_angular_momentum(
                skf_prefix, symbol, skf_sep, skf_suffix
            )

    out.append(_format_argument('hamiltonian', ('dftb', ham)))

    if 'forces' in properties or 'stress' in properties:
        if 'analysis' not in params:
            params['analysis'] = dict()
        params['analysis']['calculateforces'] = True

    for key, val in params.items():
        out.append(_format_argument(key, val))

    fd.write('\n'.join(out))


def _read_max_angular_momentum(prefix, symbol, sep, suffix):
    """Read maximum angular momentum from .skf file.

    See dftb.org for A detailed description of the Slater-Koster file format.
    """
    fname = os.path.join(prefix, '{0}{1}{0}{2}'.format(symbol, sep, suffix))
    with open(fname, 'r') as fd:
        line = fd.readline()
        if line[0] == '@':
            # Extended format
            fd.readline()
            l = 3
            pos = 9
        else:
            # Simple format:
            l = 2
            pos = 7

        # Sometimes there ar commas, sometimes not:
        line = fd.readline().replace(',', ' ')

        occs = [float(f) for f in line.split()[pos:pos + l + 1]]
        for f in occs:
            if f > 0.0:
                return 'spdf'[l]
            l -= 1


_type_to_dtype = dict(integer=int,
                      real=float,
                      complex=complex,
                      logical=bool)


@reader
def _parse_results_tag(fd):
    results = dict()
    for line in fd:
        tokens = line.split(':')
        name = tokens[0].strip()
        dtype = _type_to_dtype[tokens[1]]
        dim = int(tokens[2])
        if dim == 0:
            results[name] = dtype(fd.readline().strip())
            continue

        # reversed because fortran->C order
        shape = list(reversed([int(x) for x in tokens[3].split(',')]))

        ntot = np.prod(shape)
        data = np.zeros(ntot, dtype=dtype)
        nread = 0
        while nread < ntot:
            newdata = list(map(dtype, fd.readline().strip().split()))
            ndata = len(newdata)
            data[nread:nread + ndata] = newdata
            nread += ndata
        results[name] = data.reshape(shape)
    return results


@reader
def _parse_detailed_out(fd):
    detailed = dict()
    for line in fd:
        if line.strip() == 'Atomic gross charges (e)':
            fd.readline()
            charges = []
            while True:
                tokens = fd.readline().strip().split()
                if not tokens:
                    break
                charges.append(float(tokens[1]))
            detailed['charges'] = np.array(charges)
        elif line.startswith('Fermi level:'):
            detailed['efermi'] = float(line.strip().split()[2]) * Hartree
        elif line.startswith('Dipole moment:'):
            tokens = line.strip().split()
            if tokens[-1] == 'Debye':
                dipole = list(map(float, tokens[2:5]))
                detailed['dipole'] = np.array(dipole) * Debye
    return detailed


@reader
def _parse_band_out(fd):
    kpts = []
    for line in fd:
        tokens = line.strip().split()
        kpt = int(tokens[1]) - 1
        spin = int(tokens[3]) - 1
        weight = float(tokens[5])
        eps_n = []
        f_n = []
        while True:
            tokens = fd.readline().strip().split()
            if not tokens:
                break
            eps_n.append(float(tokens[1]))
            f_n.append(float(tokens[2]))
        kpts.append(SinglePointKPoint(weight, spin, kpt, eps_n=np.array(eps_n),
                                      f_n=np.array(f_n)))
    return kpts


@reader
def _parse_ibz_kpts(fd):
    for line in fd:
        if line.startswith('K-points and weights:'):
            ibz_kpts = [list(map(float, line.strip().split()[4:7]))]
            while True:
                tokens = fd.readline().strip().split()
                if not tokens:
                    return np.array(ibz_kpts)
                ibz_kpts.append(list(map(float, tokens[1:4])))
    else:
        return None


def get_dftb_results(atoms, dirname, label):
    results = _parse_results_tag(os.path.join(dirname, 'results.tag'))
    energy = results['extrapolated0_energy'] * Hartree
    free_energy = results['forcerelated_energy'] * Hartree
    stress = results.get('stress', None)
    if stress is not None:
        stress = -stress.flat[[0, 4, 8, 5, 2, 1]] * Hartree / Bohr**3
    forces = results.get('forces', None)
    if forces is not None:
        forces *= Hartree / Bohr

    detailed = _parse_detailed_out(os.path.join(dirname, 'detailed.out'))
    efermi = detailed.get('efermi')
    charges = detailed.get('charges')

    ibz_kpts = _parse_ibz_kpts(label + '.out')

    calc = SinglePointDFTCalculator(atoms=atoms, energy=energy, efermi=efermi,
                                    free_energy=free_energy, forces=forces,
                                    stress=stress, ibzkpts=ibz_kpts,
                                    charges=charges)
    kpts = _parse_band_out(os.path.join(dirname, 'band.out'))
    calc.kpts = kpts
    return calc
