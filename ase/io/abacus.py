# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:31:30 2018
@author: shenzx

Modified on Wed Aug 01 11:44:51 2022
@author: Ji Yu-yang
"""

import re
import warnings
import numpy as np
import os
import shutil

from ase import Atoms
from ase.units import Bohr, Hartree, GPa
from ase.utils import lazymethod, lazyproperty, reader, writer
from ase.calculators.singlepoint import SinglePointDFTCalculator, arrays_to_kpoints
_re_float = r'[-+]?\d+\.*\d*(?:[Ee][-+]\d+)?'

# --------WRITE---------

# WRITE ABACUS INPUT -START-
@writer
def write_input(fd, parameters=None):
    """Write the INPUT file for ABACUS

    Parameters
    ----------
    fd: str
        The file object to write to
    parameters: dict
        The dictionary of all paramters for the calculation. 
    """
    from copy import deepcopy
    params = deepcopy(parameters)
    params['dft_functional'] = params.pop('xc') if params.get(
        'xc') else params.get('dft_functional', 'pbe')
    for key in ['pp', 'basis', 'pseudo_dir', 'basis_dir', 'offsite_basis_dir', 'kpts']:
        params.pop(key, None)

    lines = []
    lines.append('INPUT_PARAMETERS')
    lines.append('# Created by Atomic Simulation Enviroment')
    for key, val in params.items():
        if val is not None:
            lines.append(str(key) + ' ' * (20 - len(key)) + str(val))
    lines.append('')
    fd.write('\n'.join(lines))
# WRITE ABACUS INPUT -END-

# WRITE ABACUS KPT -START-


@writer
def write_kpt(fd=None, parameters=None):
    """Write the KPT file for ABACUS

    Parameters
    ----------
    fd: str
        The file object to write to
    parameters: dict
        The dictionary of all paramters for the calculation. 
        If `gamma_only` or `kspacing` in `parameters`, it will not output any files by ase
    """

    gamma_only = parameters.get('gamma_only', 0)
    kspacing = parameters.get('kspacing', 0.0)
    if gamma_only is not None and gamma_only == 1:
        return
    if kspacing is not None and kspacing > 0.0:
        return
    else:
        lines = []
        knumber = parameters.get('knumber', 0)
        kmode = parameters.get('kmode', 'Gamma')
        kpts = parameters.get('kpts', [1, 1, 1])
        koffset = parameters.get('koffset', [0, 0, 0])
        lines.append('K_POINTS')
        lines.append(f'{knumber}')
        lines.append(f'{kmode}')
        if kmode in ['Gamma', 'MP']:
            lines.append(
                ' '.join(map(str, kpts)) + ' ' + ' '.join(map(str, koffset)))
        elif kmode in ['Direct', 'Cartesian', 'Line']:
            for n in range(len(kpts)):
                lines.append(
                    f'{kpts[n][0]:0<12f} {kpts[n][1]:0<12f} {kpts[n][2]:0<12f}')
        else:
            raise ValueError("The value of kmode is not right, set to "
                             "Gamma, MP, Direct, Cartesian, or Line.")
        lines.append('')
        fd.write('\n'.join(lines))
# WRITE ABACUS KPT -END-


def _copy_files(file_list, src, dst, env, name):
    if not src:
        # environment variable for PP paths
        if env in os.environ:
            src = os.environ[env]
        else:
            src = './'
            # raise NotFoundErr(
            #     f"Can not set directory of {name} according to environment variable {env}")

    for val in file_list:
        src_file = os.path.join(src, val.strip())
        dst_file = os.path.join(dst, val.strip())
        if os.path.exists(dst_file):
            continue
        elif os.path.exists(src_file):
            shutil.copyfile(src_file, dst_file)
        else:
            raise FileNotFoundError(
                f"Can't find {name} for ABACUS calculation")


# WRITE ABACUS PP -START-
def copy_pp(pp_list, pseudo_dir=None, directory='./'):
    """Copy pseudo-potential files from `pseudo_dir` to `directory`

    Parameters
    ----------
    pp_list: list
        List of pseudo-potential files, e.g. ['Si.UPF', 'C.UPF']
    pseudo_dir: str
        The src directory includes pseudo-potential files. If None, 
        it will get directory from environment variable `ABACUS_PP_PATH`
    directory: str
        The dst directory
    """
    _copy_files(pp_list, pseudo_dir, directory,
                'ABACUS_PP_PATH', 'pseudo-potential files')

# WRITE ABACUS PP -END-


# WRITE ABACUS basis -START-
def copy_basis(basis_list, basis_dir=None, directory='./'):
    """Copy LCAO basis files from `basis_dir` to `directory`

    Parameters
    ----------
    basis_list: list
        List of LCAO basis files, e.g. ['Si.orb', 'C.orb']
    basis_dir: str
        The src directory includes LCAO basis files. If None, 
        it will get directory from environment variable `ABACUS_ORBITAL_PATH`
    directory: str
        The dst directory
    """
    _copy_files(basis_list, basis_dir, directory,
                'ABACUS_ORBITAL_PATH', 'basis files')

# WRITE ABACUS basis -END-

# WRITE ABACUS basis -START-


def copy_offsite_basis(offsite_basis_list, offsite_basis_dir=None, directory='./'):
    """Copy off-site ABFs basis files from `basis_dir` to `directory`

    Parameters
    ----------
    offsite_basis_list: list
        List of off-site ABFs basis files, e.g. ['abfs_Si.dat', 'abfs_C.dat']
    offsite_basis_dir: str
        The src directory includes off-site ABFs basis files. If None, 
        it will get directory from environment variable `ABACUS_ABFS_PATH`
    directory: str
        The dst directory
    """
    _copy_files(offsite_basis_list, offsite_basis_dir, directory,
                'ABACUS_ABFS_PATH', 'off-site ABFs basis files')

# WRITE ABACUS basis -END-


# WRITE ABACUS STRU -START-


def judge_exist_stru(stru=None):
    if stru is None:
        return False
    else:
        return True


def read_ase_stru(stru=None, coordinates_type="Cartesian"):
    if judge_exist_stru(stru):
        atoms_list = []
        atoms_position = []
        atoms_masses = []
        atoms_magnetism = []
        atoms_all = stru.get_chemical_symbols()

        # sort atoms according to atoms
        for atoms_all_name in atoms_all:
            temp = True
            for atoms_list_name in atoms_list:
                if atoms_all_name == atoms_list_name:
                    temp = False
                    break

            if temp:
                atoms_list.append(atoms_all_name)

        for atoms_list_name in atoms_list:
            atoms_position.append([])
            atoms_masses.append([])
            atoms_magnetism.append(0)

        # get position, masses, magnetism from ase atoms
        # TODO: property 'magmoms' is not implemented in ABACUS
        if coordinates_type == 'Cartesian':
            for i in range(len(atoms_list)):
                for j in range(len(atoms_all)):
                    if atoms_all[j] == atoms_list[i]:
                        atoms_position[i].append(list(
                            stru.get_positions()[j]))
                        atoms_masses[i] = stru.get_masses()[j]
                        atoms_magnetism[i] += np.linalg.norm(stru[j].magmom)

        elif coordinates_type == 'Direct':
            for i in range(len(atoms_list)):
                for j in range(len(atoms_all)):
                    if atoms_all[j] == atoms_list[i]:
                        atoms_position[i].append(list(
                            stru.get_scaled_positions()[j]))
                        atoms_masses[i] = stru.get_masses()[j]
                        atoms_magnetism[i] += np.linalg.norm(stru[j].magmom)

        else:
            raise ValueError("'coordinates_type' is ERROR,"
                             "please set to 'Cartesian' or 'Direct'")

        return atoms_list, atoms_masses, atoms_position, atoms_magnetism


def write_input_stru_core(fd,
                          stru=None,
                          pp=None,
                          basis=None,
                          offsite_basis=None,
                          coordinates_type="Cartesian",
                          atoms_list=None,
                          atoms_position=None,
                          atoms_masses=None,
                          atoms_magnetism=None,
                          fix=None,
                          init_vel=False):
    lines = []
    if not judge_exist_stru(stru):
        return "No input structure!"

    elif (atoms_list is None):
        return "Please set right atoms list"
    elif(atoms_position is None):
        return "Please set right atoms position"
    elif(atoms_masses is None):
        return "Please set right atoms masses"
    elif(atoms_magnetism is None):
        return "Please set right atoms magnetism"
    else:
        lines.append('ATOMIC_SPECIES')
        for i, elem in enumerate(atoms_list):
            if pp:
                pseudofile = pp.get(elem, '')
            else:
                pseudofile = ''
            temp1 = ' ' * (4 - len(atoms_list[i]))
            temp2 = ' ' * (14 - len(str(atoms_masses[i])))
            atomic_species = (atoms_list[i] + temp1
                              + str(atoms_masses[i]) + temp2
                              + pseudofile)

            lines.append(atomic_species)

        if basis:
            lines.append('')
            lines.append('NUMERICAL_ORBITAL')
            for i, elem in enumerate(atoms_list):
                orbitalfile = basis[elem]
                lines.append(orbitalfile)

        if offsite_basis:
            lines.append('')
            lines.append('ABFS_ORBITAL')
            for i, elem in enumerate(atoms_list):
                orbitalfile = offsite_basis[elem]
            lines.append(orbitalfile)

        lines.append('')
        lines.append('LATTICE_CONSTANT')
        lines.append(f'{1/Bohr}')

        lines.append('')
        lines.append('LATTICE_VECTORS')
        for i in range(3):
            temp3 = f'{stru.get_cell()[i][0]:0<12f} {stru.get_cell()[i][1]:0<12f} {stru.get_cell()[i][1]:0<12f} '
            lines.append(temp3)

        lines.append('')
        lines.append('ATOMIC_POSITIONS')
        lines.append(coordinates_type)
        for i in range(len(atoms_list)):
            lines.append('')
            lines.append(atoms_list[i])
            lines.append(str("{:0<12f}".format(float(atoms_magnetism[i]))))
            lines.append(str(len(atoms_position[i])))

            for j in range(len(atoms_position[i])):
                temp4 = str("{:0<12f}".format(
                    atoms_position[i][j][0])) + ' '
                temp5 = str("{:0<12f}".format(
                    atoms_position[i][j][1])) + ' '
                temp6 = str("{:0<12f}".format(
                    atoms_position[i][j][2])) + ' '
                sym_pos = temp4 + temp5 + temp6 + \
                    f'{fix[j][0]:.0f} {fix[j][1]:.0f} {fix[j][2]:.0f} '
                if init_vel:   # velocity in unit A/fs ?
                    sym_pos += f'v {stru.get_velocities()[j][0]} {stru.get_velocities()[j][1]} {stru.get_velocities()[j][2]} '
                if stru[j].magmom:
                    if isinstance(stru[j].magmom, float):
                        if len(stru) == 1:
                            sym_pos += f'mag {stru[j].magmom} '
                        else:
                            raise ValueError(
                                "If the number of Atom objects is larger than one, magnetic moment of each atom can not be obtained by ABACUS at present. One should set `mag` flags in ABACUS 'STRU' file.")
                    elif len(stru[j].magmom) == 3:
                        sym_pos += f'mag {stru[j].magmom[0]} {stru[j].magmom[1]} {stru[j].magmom[2]} '
                    elif stru[j].magmom == 1:
                        sym_pos += f'mag {stru[j].magmom[0]} '
                lines.append(sym_pos)
    lines.append('')
    fd.write('\n'.join(lines))


@writer
def write_abacus(fd,
                 atoms=None,
                 pp=None,
                 basis=None,
                 offsite_basis=None,
                 scaled=True,
                 init_vel=False):
    """Write the STRU file for ABACUS

    Parameters
    ----------
    fd: str
        The file object to write to
    atoms: atoms.Atoms
        The Atoms object for the requested calculation
    pp: dict
        The pseudo-potential file of each elements, e.g. for SiC, {'Si':'Si.UPF', 'C':'C.UPF'}
    basis: dict
        The basis file of each elements for LCAO calculations, e.g. for SiC, {'Si':'Si.orb', 'C':'C.orb'}
    offsite_basis: dict
        The offsite basis file of each elements for HSE calculations with off-site ABFs, e.g. for SiC, {'Si':'Si.orb', 'C':'C.orb'}
    scaled: bool
        If output STRU file with scaled positions
    init_vel: bool
        if initialize velocities in STRU file
    """

    if scaled:
        coordinates_type = 'Direct'
    else:
        coordinates_type = 'Cartesian'

    if not judge_exist_stru(atoms):
        return "No input structure!"

    else:
        (atoms_list,
         atoms_masses,
         atoms_position,
         atoms_magnetism) = read_ase_stru(atoms, coordinates_type)

        from ase.constraints import FixAtoms, FixCartesian
        fix_cart = np.ones([len(atoms), 3])
        if atoms.constraints:
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    fix_cart[constr.index] = [0, 0, 0]
                elif isinstance(constr, FixCartesian):
                    fix_cart[constr.index] = constr.mask

        write_input_stru_core(fd,
                              atoms,
                              pp,
                              basis,
                              offsite_basis,
                              coordinates_type,
                              atoms_list,
                              atoms_position,
                              atoms_masses,
                              atoms_magnetism,
                              fix_cart,
                              init_vel,
                              )
# WRITE ABACUS STRU -END-

# --------READ---------

# Read KPT file  -START-
@reader
def read_kpt(fd):
    """Read ABACUS KPT file and return results dict"""
    lines = fd.readlines()
    kmode = None
    knumber = None
    kpts = None
    koffset = None
    if lines[2][-1] == '\n':
        kmode = lines[2][:-1]
    else:
        kmode = lines[2]
    if kmode in ['Gamma', 'MP']:
        knumber = lines[1].split()[0]
        kpts = np.array(lines[3].split()[:3])
        koffset = np.array(lines[3].split()[3:])
    elif kmode in ['Cartesian', 'Direct', 'Line']:
        knumber = lines[1].split()[0]
        kpts = np.array([list(map(float, line.split()[:3]))
                         for line in lines[3:]])
    else:
        raise ValueError("The value of kmode is not right, set to "
                         "Gamma, MP, Direct, Cartesian, or Line.")

    return {'mode': kmode,
            'number': knumber,
            'kpts': kpts,
            'offset': koffset}
# Read KPT file  -END-

# READ ABACUS STRU -START-
@reader
def read_abacus(fd, latname=None, verbose=False):
    """Read structure information from abacus structure file.

    If `latname` is not None, 'LATTICE_VECTORS' should be removed in structure files of ABACUS.
    Allowed values: 'sc', 'fcc', 'bcc', 'hexagonal', 'trigonal', 'st', 'bct', 'so', 'baco', 'fco', 'bco', 'sm', 'bacm', 'triclinic'

    If `verbose` is True, pseudo-potential, basis and other information along with the Atoms object will be output as a dict.
    """

    from ase.constraints import FixCartesian

    contents = fd.read()
    title_str = r'(?:LATTICE_CONSTANT|NUMERICAL_DESCRIPTOR|NUMERICAL_ORBITAL|ABFS_ORBITAL|LATTICE_VECTORS|LATTICE_PARAMETERS|ATOMIC_POSITIONS)'

    # remove comments and empty lines
    contents = re.compile(r"#.*|//.*").sub('', contents)
    contents = re.compile(r'\n{2,}').sub('\n', contents)

    # specie, mass, pps
    specie_pattern = re.compile(
        rf'ATOMIC_SPECIES\s*\n([\s\S]+?)\s*\n{title_str}')
    specie_lines = np.array(
        [line.split() for line in specie_pattern.search(contents).group(1).split('\n')])
    symbols = specie_lines[:, 0]
    ntype = len(symbols)
    atom_mass = specie_lines[:, 1].astype(float)
    try:
        atom_potential = dict(zip(symbols, specie_lines[:, 2].tolist()))
    except IndexError:
        atom_potential = None

    # basis
    aim_title = 'NUMERICAL_ORBITAL'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    orb_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    orb_lines = orb_pattern.search(contents)
    if orb_lines:
        atom_basis = dict(zip(symbols, orb_lines.group(1).split('\n')))
    else:
        atom_basis = None

    # ABFs basis
    aim_title = 'ABFS_ORBITAL'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    abf_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    abf_lines = abf_pattern.search(contents)
    if abf_lines:
        atom_offsite_basis = dict(zip(symbols, abf_lines.group(1).split('\n')))
    else:
        atom_offsite_basis = None

    # deepks for ABACUS
    aim_title = 'NUMERICAL_DESCRIPTOR'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    deep_pattern = re.compile(
        rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    deep_lines = deep_pattern.search(contents)
    if deep_lines:
        atom_descriptor = deep_lines.group(1)
    else:
        atom_descriptor = None

    # lattice constant
    aim_title = 'LATTICE_CONSTANT'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    a0_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    a0_lines = a0_pattern.search(contents)
    atom_lattice_scale = float(a0_lines.group(1))

    # lattice vector
    if latname:
        aim_title = 'LATTICE_PARAMETERS'
        aim_title_sub = title_str.replace('|' + aim_title, '')
        lparam_pattern = re.compile(
            rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
        lparam_lines = lparam_pattern.search(contents)
        atom_lattice = get_lattice_from_latname(lparam_lines, latname)
    else:
        aim_title = 'LATTICE_VECTORS'
        aim_title_sub = title_str.replace('|' + aim_title, '')
        vec_pattern = re.compile(
            rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
        vec_lines = vec_pattern.search(contents)
        if vec_lines:
            atom_lattice = np.array([line.split() for line in vec_pattern.search(
                contents).group(1).split('\n')]).astype(float)
        else:
            raise Exception(
                f"Parameter `latname` or `LATTICE_VECTORS` in {fd.name} must be set.")
    atom_lattice = atom_lattice * atom_lattice_scale * Bohr

    aim_title = 'ATOMIC_POSITIONS'
    type_pattern = re.compile(rf'{aim_title}\s*\n(\w+)\s*\n')
    # type of coordinates
    atom_pos_type = type_pattern.search(contents).group(1)
    assert atom_pos_type in [
        'Direct', 'Cartesian'], "Only two type of atomic coordinates are supported: 'Direct' or 'Cartesian'."

    block_pattern = re.compile(rf'{atom_pos_type}\s*\n([\s\S]+)')
    block = block_pattern.search(contents).group()
    atom_magnetism = []
    atom_symbol = []
    atom_block = []
    for i, symbol in enumerate(symbols):
        pattern = re.compile(rf'{symbol}\s*\n({_re_float})\s*\n(\d+)')
        sub_block = pattern.search(block)
        number = int(sub_block.group(2))

        # symbols, magnetism
        sym = [symbol] * number
        atom_mags = [float(sub_block.group(1))] * number
        for j in range(number):
            atom_symbol.append(sym[j])
            atom_magnetism.append(atom_mags[j])

        if i == ntype - 1:
            lines_pattern = re.compile(
                rf'{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+)\s*\n')
        else:
            lines_pattern = re.compile(
                rf'{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+?)\s*\n\w+\s*\n{_re_float}')
        lines = lines_pattern.search(block)
        for j in [line.split() for line in lines.group(1).split('\n')]:
            atom_block.append(j)
    atom_block = np.array(atom_block)
    atom_magnetism = np.array(atom_magnetism)

    # position
    atom_positions = atom_block[:, 0:3].astype(float)
    natoms = len(atom_positions)

    # fix_cart
    if (atom_block[:, 3] == ['m'] * natoms).all():
        atom_xyz = ~atom_block[:, 4:7].astype(bool)
    else:
        atom_xyz = ~atom_block[:, 3:6].astype(bool)
    fix_cart = [FixCartesian(ci, xyz) for ci, xyz in enumerate(atom_xyz)]

    def _get_index(labels, num):
        index = None
        res = []
        for l in labels:
            if l in atom_block:
                index = np.where(atom_block == l)[-1][0]
        if index is not None:
            res = atom_block[:, index + 1:index + 1 + num].astype(float)

        return res, index

    # velocity
    v_labels = ['v', 'vel', 'velocity']
    atom_vel, v_index = _get_index(v_labels, 3)

    # magnetism
    m_labels = ['mag', 'magmom']
    if 'angle1' in atom_block or 'angle2' in atom_block:
        warnings.warn(
            "Non-colinear angle-settings are not yet supported for this interface.")
    mags, m_index = _get_index(m_labels, 1)
    try:     # non-colinear
        if m_index:
            atom_magnetism = atom_block[:,
                                        m_index + 1:m_index + 4].astype(float)
    except IndexError:  # colinear
        if m_index:
            atom_magnetism = mags

    # to ase
    if atom_pos_type == 'Direct':
        atoms = Atoms(symbols=atom_symbol,
                      cell=atom_lattice,
                      scaled_positions=atom_positions,
                      pbc=True)
    elif atom_pos_type == 'Cartesian':
        atoms = Atoms(symbols=atom_symbol,
                      cell=atom_lattice,
                      positions=atom_positions * atom_lattice_scale * Bohr,
                      pbc=True)
    if v_index:
        atoms.set_velocities(atom_vel)

    atoms.set_initial_magnetic_moments(atom_magnetism)
    atoms.set_constraint(fix_cart)

    if verbose:
        return {
            "atoms": atoms,
            "pp": atom_potential,
            "basis": atom_basis,
            "offsite_basis": atom_offsite_basis,
            "descriptor": atom_descriptor
        }
    else:
        return atoms


def get_lattice_from_latname(lines, latname=None):

    from math import sqrt
    if lines:
        lines = lines.group(1).split(' ')

    if latname == 'sc':
        return np.eye(3)
    elif latname == 'fcc':
        return np.array([[-0.5, 0, 0.5], [0, 0.5, 0.5], [-0.5, 0.5, 0]])
    elif latname == 'bcc':
        return np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]])
    elif latname == 'hexagonal':
        x = float(lines[0])
        return np.array([[1.0, 0, 0], [-0.5, sqrt(3) / 2, 0], [0, 0, x]])
    elif latname == 'trigonal':
        x = float(lines[0])
        tx = sqrt((1 - x) / 2)
        ty = sqrt((1 - x) / 6)
        tz = sqrt((1 + 2 * x) / 3)
        return np.array([[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]])
    elif latname == 'st':
        x = float(lines[0])
        return np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, x]])
    elif latname == 'bct':
        x = float(lines[0])
        return np.array([[0.5, -0.5, x], [0.5, 0.5, x], [0.5, 0.5, x]])
    elif latname == 'baco':
        x, y = list(map(float, lines))
        return np.array([[0.5, x / 2, 0], [-0.5, x / 2, 0], [0, 0, y]])
    elif latname == 'fco':
        x, y = list(map(float, lines))
        return np.array([[0.5, 0, y / 2], [0.5, x / 2, 0], [0.5, x / 2, 0]])
    elif latname == 'bco':
        x, y = list(map(float, lines))
        return np.array([[0.5, x / 2, y / 2], [-0.5, x / 2, y / 2], [-0.5, -x / 2, y / 2]])
    elif latname == 'bco':
        x, y, z = list(map(float, lines))
        return np.array([[1, 0, 0], [x * z, x * sqrt(1 - z**2), 0], [0, 0, y]])
    elif latname == 'bacm':
        x, y, z = list(map(float, lines))
        return np.array([[0.5, 0, -y / 2], [x * z, x * sqrt(1 - z**2), 0], [0.5, 0, y / 2]])
    elif latname == 'triclinic':
        x, y, m, n, l = list(map(float, lines))
        fac = sqrt(1 + 2 * m * n * l - m**2 - n**2 - l**2) / sqrt(1 - m**2)
        return np.array([[1, 0, 0], [x * m, x * sqrt(1 - m**2), 0], [y * n, y * (l - n * m / sqrt(1 - m**2)), y * fac]])
# READ ABACUS STRU -END-


# READ ABACUS OUT -START-
class AbacusOutChunk:
    """Base class for AbacusOutChunks"""

    def __init__(self, contents):
        """Constructor

        Parameters
        ----------
        contents: str
            The contents of the output file
        """
        self.contents = contents

    def parse_scalar(self, pattern):
        """Parse a scalar property from the chunk according to specific pattern

        Parameters
        ----------
        pattern: str
            The pattern used to parse

        Returns
        -------
        float
            The scalar value of the property
        """
        pattern_compile = re.compile(pattern)
        res = pattern_compile.search(self.contents)
        if res:
            return float(res.group(1))
        else:
            return None

    @lazyproperty
    def coordinate_system(self):
        """Parse coordinate system (Cartesian or Direct) from the output file"""
        # for '|', it will match all the patterns which results in '' or None
        class_pattern = re.compile(
            r'(DIRECT) COORDINATES|(CARTESIAN) COORDINATES')
        coord_class = list(class_pattern.search(self.contents).groups())
        _remove_empty(coord_class)

        return coord_class[0]

    @lazymethod
    def _parse_site(self):
        """Parse sites for all the structures in the output file"""
        pos_pattern = re.compile(
            rf'(CARTESIAN COORDINATES \( UNIT = {_re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n')

        return pos_pattern.findall(self.contents)


class AbacusOutHeaderChunk(AbacusOutChunk):
    """General information that the header of the running_*.log file contains"""

    def __init__(self, contents):
        """Constructor

        Parameters
        ----------
        contents: str
            The contents of the output file
        """
        super().__init__(contents)

    @lazyproperty
    def out_dir(self):
        out_pattern = re.compile(r"global_out_dir\s*=\s*([\s\S]+?)/")
        return out_pattern.search(self.contents).group(1)

    @lazyproperty
    def lattice_constant(self):
        """The lattice constant from the header of the running_*.log"""
        a0_pattern_str = rf'lattice constant \(Angstrom\)\s*=\s*({_re_float})'
        return self.parse_scalar(a0_pattern_str)

    @lazyproperty
    def initial_cell(self):
        """The initial cell from the header of the running_*.log file"""
        cell_pattern = re.compile(
            rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')
        lattice = np.reshape(cell_pattern.findall(
            self.contents)[0], (3, 3)).astype(float)

        return lattice * self.lattice_constant

    @lazyproperty
    def initial_site(self):
        def str_to_sites(val_in):
            val = np.array(val_in)
            labels = val[:, 0]
            pos = val[:, 1:4].astype(float)
            if val.shape[1] == 5:
                mag = val[:, 4].astype(float)
                vel = np.zeros((3, ), dtype=float)
            elif val.shape[1] == 8:
                mag = val[:, 4].astype(float)
                vel = val[:, 5:8].astype(float)
            return labels, pos, mag, vel

        def parse_block(pos_block):
            data = list(pos_block)
            _remove_empty(data)
            site = list(map(list, site_pattern.findall(data[0])))
            list(map(_remove_empty, site))
            labels, pos, mag, vel = str_to_sites(site)
            if self.coordinate_system == 'CARTESIAN':
                unit = float(unit_pattern.search(self.contents).group(1)) * Bohr
                positions = pos * unit
            elif self.coordinate_system == 'DIRECT':
                positions = pos
            return labels, positions, mag, vel

        site_pattern = re.compile(
            rf'tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})|tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})')
        unit_pattern = re.compile(rf'UNIT = ({_re_float}) Bohr')

        return parse_block(self._parse_site()[0])

    @lazyproperty
    def initial_atoms(self):
        """Create an atoms object for the initial structure from the
        header of the running_*.log file"""
        labels, positions, mag, vel = self.initial_site
        if self.coordinate_system == 'CARTESIAN':
            atoms = Atoms(symbols=labels, positions=positions,
                          cell=self.initial_cell, pbc=True, velocities=vel)
        elif self.coordinate_system == 'DIRECT':
            atoms = Atoms(symbols=labels, scaled_positions=positions,
                          cell=self.initial_cell, pbc=True, velocities=vel)
        atoms.set_initial_magnetic_moments(mag)

        return atoms

    @lazyproperty
    def is_relaxation(self):
        """Determine if the calculation is an atomic position optimization or not"""
        return 'STEP OF ION RELAXATION' in self.contents

    @lazyproperty
    def is_cell_relaxation(self):
        """Determine if the calculation is an variable cell optimization or not"""
        return 'RELAX CELL' in self.contents

    @lazyproperty
    def is_md(self):
        """Determine if calculation is a molecular dynamics calculation"""
        return 'STEP OF MOLECULAR DYNAMICS' in self.contents

    @lazymethod
    def _parse_k_points(self):
        """Get the list of k-points used in the calculation"""
        def str_to_kpoints(val_in):
            lines = re.search(
                rf'KPOINTS\s*DIRECT_X\s*DIRECT_Y\s*DIRECT_Z\s*WEIGHT([\s\S]+?)DONE', val_in).group(1).strip().split('\n')
            data = []
            for line in lines:
                data.append(line.strip().split()[1:5])
            data = np.array(data, dtype=float)
            kpoints = data[:, :3]
            weights = data[:, 3]
            return kpoints, weights

        k_pattern = re.compile(
            r'minimum distributed K point number\s*=\s*\d+([\s\S]+?DONE : INIT K-POINTS Time)')
        sub_contents = k_pattern.search(self.contents).group(1)
        k_points, k_point_weights = str_to_kpoints(sub_contents)

        return k_points[:int(self.n_k_points)], k_point_weights[:int(self.n_k_points)]

    @lazyproperty
    def n_atoms(self):
        """The number of atoms for the material"""
        pattern_str = r'TOTAL ATOM NUMBER = (\d+)'

        return int(self.parse_scalar(pattern_str))

    @lazyproperty
    def n_bands(self):
        """The number of Kohn-Sham states for the chunk"""
        pattern_str = r'NBANDS = (\d+)'

        return int(self.parse_scalar(pattern_str))

    @lazyproperty
    def n_occupied_bands(self):
        """The number of occupied Kohn-Sham states for the chunk"""
        pattern_str = r'occupied bands = (\d+)'

        return int(self.parse_scalar(pattern_str))

    @lazyproperty
    def n_spins(self):
        """The number of spin channels for the chunk"""
        pattern_str = r'nspin = (\d+)'

        return 1 if int(self.parse_scalar(pattern_str)) in [1, 4] else 2

    @lazyproperty
    def n_k_points(self):
        """The number of spin channels for the chunk"""
        nks = self.parse_scalar(r'nkstot_ibz = (\d+)') if self.parse_scalar(
            r'nkstot_ibz = (\d+)') else self.parse_scalar(r'nkstot = (\d+)')

        return int(nks)

    @lazyproperty
    def k_points(self):
        """All k-points listed in the calculation"""
        return self._parse_k_points()[0]

    @lazyproperty
    def k_point_weights(self):
        """The k-point weights for the calculation"""
        return self._parse_k_points()[1]

    @lazyproperty
    def header_summary(self):
        """Dictionary summarizing the information inside the header"""
        return {
            "lattice_constant": self.lattice_constant,
            "initial_atoms": self.initial_atoms,
            "initial_cell": self.initial_cell,
            "is_relaxation": self.is_relaxation,
            "is_cell_relaxation": self.is_cell_relaxation,
            "is_md": self.is_md,
            "n_atoms": self.n_atoms,
            "n_bands": self.n_bands,
            "n_occupied_bands": self.n_occupied_bands,
            "n_spins": self.n_spins,
            "n_k_points": self.n_k_points,
            "k_points": self.k_points,
            "k_point_weights": self.k_point_weights,
            "out_dir": self.out_dir
        }


class AbacusOutCalcChunk(AbacusOutChunk):
    """A part of the running_*.log file correponding to a single calculated structure"""

    def __init__(self, contents, header, index=-1):
        """Constructor

        Parameters
        ----------
        lines: str
            The contents of the output file
        header: dict
            A summary of the relevant information from the running_*.log header
        index: slice or int
            index of image. `index = 0` is the first calculated image rather initial image
        """
        super().__init__(contents)
        self._header = header.header_summary
        self.index = index

    @lazymethod
    def _parse_cells(self):
        """Parse all the cells from the output file"""
        if self._header['is_relaxation']:
            return [self.initial_cell for i in range(self.ion_steps)]
        elif self._header['is_cell_relaxation']:
            cell_pattern = re.compile(
                rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')
            _lattice = np.reshape(cell_pattern.findall(
                self.contents), (-1, 3, 3)).astype(float)
            if self.ion_steps and _lattice.shape[0] != self.ion_steps:
                lattice = np.zeros((self.ion_steps, 3, 3), dtype=float)
                _indices = np.where(self._parse_relaxation_convergency())[0]
                for i in range(len(_indices)):
                    if i == 0:
                        lattice[:_indices[i] + 1] = self.initial_cell
                    else:
                        lattice[_indices[i - 1] +
                                1:_indices[i] + 1] = _lattice[i - 1]
            return lattice * self._header['lattice_constant']
        else:
            return self.initial_cell

    @lazyproperty
    def ion_steps(self):
        "The number of ion steps"
        return len(self._parse_ionic_block())

    @lazymethod
    def _parse_forces(self):
        """Parse all the forces from the output file"""
        force_pattern = re.compile(
            r'TOTAL\-FORCE\s*\(eV/Angstrom\)\n\n.*\s*atom\s*x\s*y\s*z\n([\s\S]+?)\n\n')

        return force_pattern.findall(self.contents)

    @lazymethod
    def _parse_stress(self):
        """Parse the stress from the output file"""
        stress_pattern = re.compile(
            rf'(?:TOTAL\-|MD\s*)STRESS\s*\(KBAR\)\n\n.*\n\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')

        return stress_pattern.findall(self.contents)

    @lazymethod
    def _parse_eigenvalues(self):
        """Parse the eigenvalues and occupations of the system."""
        scf_eig_pattern = re.compile(
            r'(STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*\d+[\s\S]+?(?:\n\n\s*EFERMI|\n\n\n))')
        scf_eig_all = scf_eig_pattern.findall(self.contents)

        nscf_eig_pattern = re.compile(
            r'(band eigenvalue in this processor \(eV\)\s*:\n[\s\S]+?\n\n\n)')
        nscf_eig_all = nscf_eig_pattern.findall(self.contents)

        return {'scf': scf_eig_all, 'nscf': nscf_eig_all}

    @lazymethod
    def _parse_energy(self):
        """Parse the energy from the output file."""
        _out_dir = self._header['out_dir'].strip('/')
        energy_pattern = re.compile(
            rf'{_out_dir}\/\s*final etot is\s*({_re_float})\s*eV') if 'STEP OF ION RELAXATION' in self.contents or 'RELAX CELL' in self.contents else re.compile(rf'\s*final etot is\s*({_re_float})\s*eV')

        return energy_pattern.findall(self.contents)

    @lazymethod
    def _parse_efermi(self):
        """Parse the Fermi energy from the output file."""
        fermi_pattern = re.compile(rf'EFERMI\s*=\s*({_re_float})\s*eV')

        return fermi_pattern.findall(self.contents)

    @lazymethod
    def _parse_ionic_block(self):
        """Parse the ionic block from the output file"""
        step_pattern = re.compile(
            rf"(?:[NON]*SELF-|STEP OF|RELAX CELL)([\s\S]+?)charge density convergence is achieved")

        return step_pattern.findall(self.contents)

    @lazymethod
    def _parse_relaxation_convergency(self):
        """Parse the convergency of atomic position optimization from the output file"""
        pattern = re.compile(
            r"Ion relaxation is converged!|Ion relaxation is not converged yet")

        return np.array(pattern.findall(self.contents)) == "Ion relaxation is converged!"

    @lazymethod
    def _parse_cell_relaxation_convergency(self):
        """Parse the convergency of variable cell optimization from the output file"""
        pattern = re.compile(
            r"Lattice relaxation is converged!|Lattice relaxation is not converged yet")
        lat_arr = np.array(pattern.findall(self.contents)
                           ) == "Lattice relaxation is converged!"
        res = np.zeros((self.ion_steps), dtype=bool)
        if lat_arr[-1] == True:
            res[-1] = 1

        return res.astype(bool)

    @lazymethod
    def _parse_md(self):
        """Parse the molecular dynamics information from the output file"""
        md_pattern = re.compile(
            rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*({_re_float})\s*({_re_float})')

        return md_pattern.findall(self.contents)

    @lazymethod
    def get_site(self):
        """Get site from the output file according to index"""
        def str_to_sites(val_in):
            val = np.array(val_in)
            labels = val[:, 0]
            pos = val[:, 1:4].astype(float)
            if val.shape[1] == 5:
                mag = val[:, 4].astype(float)
                vel = np.zeros((3, ), dtype=float)
            elif val.shape[1] == 8:
                mag = val[:, 4].astype(float)
                vel = val[:, 5:8].astype(float)
            return labels, pos, mag, vel

        def parse_block(pos_block):
            data = list(pos_block)
            _remove_empty(data)
            site = list(map(list, site_pattern.findall(data[0])))
            list(map(_remove_empty, site))
            labels, pos, mag, vel = str_to_sites(site)
            if self.coordinate_system == 'CARTESIAN':
                unit = float(unit_pattern.search(self.contents).group(1)) * Bohr
                positions = pos * unit
            elif self.coordinate_system == 'DIRECT':
                positions = pos
            return labels, positions, mag, vel

        site_pattern = re.compile(
            rf'tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})|tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})')
        unit_pattern = re.compile(rf'UNIT = ({_re_float}) Bohr')

        return parse_block(self._parse_site()[self.index])

    @lazymethod
    def get_forces(self):
        """Get forces from the output file according to index"""
        def str_to_force(val_in):
            data = []
            val = [v.strip().split() for v in val_in.split('\n')]
            for v in val:
                data.append(np.array(v[1:], dtype=float))
            return np.array(data)

        try:
            forces = self._parse_forces()[self.index]
            return str_to_force(forces)
        except IndexError:
            return

    @lazymethod
    def get_stress(self):
        """Get the stress from the output file according to index"""
        from ase.stress import full_3x3_to_voigt_6_stress

        try:
            stress = -0.1 * GPa * \
                np.array(self._parse_stress()[self.index]).reshape(
                    (3, 3)).astype(float)
            return full_3x3_to_voigt_6_stress(stress)
        except IndexError:
            return

    @lazymethod
    def get_eigenvalues(self):
        """Get the eigenvalues and occupations of the system according to index."""
        # SCF
        def str_to_energy_occupation(val_in):
            def extract_data(val):
                def func(i):
                    res = np.array(list(map(lambda x: x.strip().split(), re.search(
                        rf'{i+1}/{nks} kpoint \(Cartesian\)\s*=.*\n([\s\S]+?)\n\n', val).group(1).split('\n'))), dtype=float)
                    return res[:, 1].astype(float), res[:, 2].astype(float)

                return np.asarray(list(map(func, [i for i in range(nks)])))

            nspin = int(re.search(
                r'STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*(\d+)', val_in).group(1))
            nks = int(
                re.search(r'\d+/(\d+) kpoint \(Cartesian\)', val_in).group(1))
            eigenvalues = np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan)
            occupations = np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan)
            if nspin in [1, 4]:
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues[0] = energies
                occupations[0] = occs
            elif nspin == 2:
                val_up = re.search(
                    r'SPIN UP :([\s\S]+?)\n\nSPIN', val_in).group()
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_up)[:, 1, :]
                eigenvalues[0] = energies
                occupations[0] = occs

                val_dw = re.search(
                    r'SPIN DOWN :([\s\S]+?)(?:\n\n\s*EFERMI|\n\n\n)', val_in).group()
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_dw)[:, 1, :]
                eigenvalues[1] = energies
                occupations[1] = occs
            return eigenvalues, occupations

        # NSCF
        def str_to_bandstructure(val_in):
            def extract_data(val):
                def func(i):
                    res = np.array(list(map(lambda x: x.strip().split(), re.search(
                        rf'k\-points{i+1}\(\d+\):.*\n([\s\S]+?)\n\n', val).group(1).split('\n'))))
                    return res[:, 2].astype(float), res[:, 3].astype(float)

                return np.asarray(list(map(func, [i for i in range(nks)])))

            nks = int(re.search(r'k\-points\d+\((\d+)\)', val_in).group(1))
            eigenvalues = np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan)
            occupations = np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan)
            if re.search('spin up', val_in) and re.search('spin down', val_in):
                val = re.search(r'spin up :\n([\s\S]+?)\n\n\n', val_in).group()
                energies, occs = extract_data(
                    val)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues[0] = energies[:int(nks / 2)]
                eigenvalues[1] = energies[int(nks / 2):]
                occupations[0] = occs[:int(nks / 2)]
                occupations[1] = occs[int(nks / 2):]
            else:
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues[0] = energies
                occupations[0] = occs
            return eigenvalues, occupations

        try:
            return str_to_energy_occupation(self._parse_eigenvalues()['scf'][self.index])
        except KeyError:
            return str_to_bandstructure(self._parse_eigenvalues()['nscf'][self.index])
        except IndexError:
            return np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan), np.full(
                (self._header['n_spins'], self._header['n_k_points'], self._header['n_bands']), np.nan)

    @lazymethod
    def get_energy(self):
        """Get the energy from the output file according to index."""
        try:
            return float(self._parse_energy()[self.index])
        except IndexError:
            return

    @lazymethod
    def get_efermi(self):
        """Get the Fermi energy from the output file according to index."""
        try:
            return float(self._parse_efermi()[self.index])
        except IndexError:
            return

    @lazymethod
    def get_relaxation_convergency(self):
        """Get the convergency of atomic position optimization from the output file"""
        return self._parse_relaxation_convergency()[self.index]

    @lazymethod
    def get_cell_relaxation_convergency(self):
        """Get the convergency of variable cell optimization from the output file"""
        return self._parse_cell_relaxation_convergency()[self.index]

    @lazymethod
    def get_md_energy(self):
        """Get the total energy of each md step"""

        try:
            return float(self._parse_md()[self.index][0]) * Hartree
        except IndexError:
            return

    @lazymethod
    def get_md_potential(self):
        """Get the potential energy of each md step"""

        try:
            return float(self._parse_md()[self.index][1]) * Hartree
        except IndexError:
            return

    @lazymethod
    def get_md_steps(self):
        """Get steps of molecular dynamics"""
        step_pattern = re.compile(r"STEP OF MOLECULAR DYNAMICS\s*:\s*(\d+)")

        return list(map(int, step_pattern.findall(self.contents)))

    @lazyproperty
    def forces(self):
        """The forces for the chunk"""
        return self.get_forces()

    @lazyproperty
    def stress(self):
        """The stress for the chunk"""
        return self.get_stress()

    @lazyproperty
    def energy(self):
        """The energy for the chunk"""
        if self._header["is_md"]:
            return self.get_md_energy()
        else:
            return self.get_energy()

    @lazyproperty
    def free_energy(self):
        """The free energy for the chunk"""
        if self._header["is_md"]:
            return self.get_md_potential()
        else:
            return self.get_energy()

    @lazyproperty
    def eigenvalues(self):
        """The eigenvalues for the chunk"""
        return self.get_eigenvalues()[0]

    @lazyproperty
    def occupations(self):
        """The occupations for the chunk"""
        return self.get_eigenvalues()[1]

    @lazyproperty
    def kpts(self):
        """The SinglePointKPoint objects for the chunk"""
        return arrays_to_kpoints(self.eigenvalues, self.occupations, self._header["k_point_weights"])

    @lazyproperty
    def E_f(self):
        """The Fermi energy for the chunk"""
        return self.get_efermi()

    @lazyproperty
    def _ionic_block(self):
        """The ionic block for the chunk"""

        return self._parse_ionic_block()[self.index]

    @lazyproperty
    def magmom(self):
        """The Fermi energy for the chunk"""
        magmom_pattern = re.compile(
            rf"total magnetism \(Bohr mag/cell\)\s*=\s*({_re_float})")

        try:
            return float(magmom_pattern.findall(self._ionic_block)[-1])
        except IndexError:
            return

    @lazyproperty
    def n_iter(self):
        """The number of SCF iterations needed to converge the SCF cycle for the chunk"""
        step_pattern = re.compile(rf"ELEC\s*=\s*[+]?(\d+)")

        try:
            return int(step_pattern.findall(self._ionic_block)[-1])
        except IndexError:
            return

    @lazyproperty
    def converged(self):
        """True if the chunk is a fully converged final structure"""
        if self._header["is_cell_relaxation"]:
            return self.get_cell_relaxation_convergency()
        elif self._header["is_relaxation"]:
            return self.get_relaxation_convergency()
        else:
            return "charge density convergence is achieved" in self.contents

    @lazyproperty
    def initial_atoms(self):
        """The initial structure defined in the running_*.log file"""
        return self._header["initial_atoms"]

    @lazyproperty
    def initial_cell(self):
        """The initial lattice vectors defined in the running_*.log file"""
        return self._header["initial_cell"]

    @lazyproperty
    def n_atoms(self):
        """The number of atoms for the material"""
        return self._header["n_atoms"]

    @lazyproperty
    def n_bands(self):
        """The number of Kohn-Sham states for the chunk"""
        return self._header["n_bands"]

    @lazyproperty
    def n_occupied_bands(self):
        """The number of occupied Kohn-Sham states for the chunk"""
        return self._header["n_occupied_bands"]

    @lazyproperty
    def n_spins(self):
        """The number of spin channels for the chunk"""
        return self._header["n_spins"]

    @lazyproperty
    def n_k_points(self):
        """The number of k_points for the chunk"""
        return self._header["n_k_points"]

    @lazyproperty
    def k_points(self):
        """k_points for the chunk"""
        return self._header["k_points"]

    @lazyproperty
    def k_point_weights(self):
        """k_point_weights for the chunk"""
        return self._header["k_point_weights"]

    @property
    def results(self):
        """Convert an AbacusOutChunk to a Results Dictionary"""
        results = {
            "energy": self.energy,
            "free_energy": self.free_energy,
            "forces": self.forces,
            "stress": self.stress,
            "magmom": self.magmom,
            "fermi_energy": self.E_f,
            "n_iter": self.n_iter,
            "eigenvalues": self.eigenvalues,
            "occupations": self.occupations
        }

        return {
            key: value for key,
            value in results.items() if value is not None}

    @lazyproperty
    def atoms(self):
        """Convert AbacusOutChunk to Atoms object and add all non-standard outputs to atoms.info"""
        """Create an atoms object for the subsequent structures
        calculated in the output file"""
        atoms = None
        if self._header['is_md']:
            from pathlib import Path

            _stru_dir = Path(self._header['out_dir']) / 'STRU'
            md_stru_dir = _stru_dir if _stru_dir.exists(
            ) else Path(self._header['out_dir'])
            atoms = read_abacus(
                open(md_stru_dir / f'STRU_MD_{self.get_md_steps()[self.index]}', 'r'))

        elif self._header['is_relaxation'] or self._header['is_cell_relaxation']:
            labels, positions, mag, vel = self.get_site()
            if self.coordinate_system == 'CARTESIAN':
                atoms = Atoms(symbols=labels, positions=positions,
                              cell=self._parse_cells()[self.index], pbc=True, velocities=vel)
            elif self.coordinate_system == 'DIRECT':
                atoms = Atoms(symbols=labels, scaled_positions=positions,
                              cell=self._parse_cells()[self.index], pbc=True, velocities=vel)

        else:
            atoms = self.initial_atoms.copy()

        calc = SinglePointDFTCalculator(
            atoms,
            energy=self.energy,
            free_energy=self.free_energy,
            forces=self.forces,
            stress=self.stress,
            magmom=self.magmom,
            ibzkpts=self.k_points,
            kpts=self.kpts)

        calc.name = 'Abacus'
        atoms.calc = calc

        return atoms


def _slice2indices(s, n=None):
    """Convert a slice object into indices"""
    if isinstance(s, slice):
        return range(*s.indices(n))
    elif isinstance(s, int):
        return [s]
    elif isinstance(s, list):
        return s
    else:
        raise ValueError(
            "Indices must be scalar integer, list of integers, or slice object")


@reader
def _get_abacus_chunks(fd, index=-1, non_convergence_ok=False):
    """Import ABACUS output files with all data available, i.e.
    relaxations, MD information, force information ..."""
    contents = fd.read()
    header_pattern = re.compile(
        r'READING GENERAL INFORMATION([\s\S]+?([NON]*SELF-|STEP OF|RELAX CELL))')
    header_chunk = AbacusOutHeaderChunk(
        header_pattern.search(contents).group(1))

    calc_pattern = re.compile(
        r'(([NON]*SELF-|STEP OF|RELAX CELL)[\s\S]+?)Total\s*Time')
    calc_contents = calc_pattern.search(contents).group(1)
    final_chunk = AbacusOutCalcChunk(calc_contents, header_chunk, -1)

    if not non_convergence_ok and not final_chunk.converged:
        raise ValueError("The calculation did not complete successfully")

    _steps = final_chunk.ion_steps if final_chunk.ion_steps else 1
    indices = _slice2indices(index, _steps)

    return [AbacusOutCalcChunk(calc_contents, header_chunk, i)
            for i in indices]


@reader
def read_abacus_out(fd, index=-1, non_convergence_ok=False):
    """Import ABACUS output files with all data available, i.e.
    relaxations, MD information, force information ..."""
    chunks = _get_abacus_chunks(fd, index, non_convergence_ok)

    return [chunk.atoms for chunk in chunks]


@reader
def read_abacus_results(fd, index=-1, non_convergence_ok=False):
    """Import ABACUS output files and summarize all relevant information
    into a dictionary"""
    chunks = _get_abacus_chunks(fd, index, non_convergence_ok)

    return [chunk.results for chunk in chunks]


def _remove_empty(a: list):
    """Remove '' and [] in `a`"""
    while '' in a:
        a.remove('')
    while [] in a:
        a.remove([])
    while None in a:
        a.remove(None)
# READ ABACUS OUT -END-
