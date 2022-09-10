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
from collections import namedtuple

from ase import Atoms
from ase.cell import Cell
from ase.units import Bohr, Hartree, GPa
from ase.utils import lazymethod, lazyproperty, reader, writer
from ase.calculators.singlepoint import SinglePointDFTCalculator, arrays_to_kpoints
_re_float = r'[-+]?\d+\.*\d*(?:[Ee][-+]\d+)?'


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
        if coordinates_type == 'Cartesian':
            for i in range(len(atoms_list)):
                for j in range(len(atoms_all)):
                    if atoms_all[j] == atoms_list[i]:
                        atoms_position[i].append(list(
                            stru.get_positions()[j]))
                        atoms_masses[i] = stru.get_masses()[j]
                        atoms_magnetism[i] += np.linalg.norm(
                            stru.get_initial_magnetic_moments()[j])

        elif coordinates_type == 'Direct':
            for i in range(len(atoms_list)):
                for j in range(len(atoms_all)):
                    if atoms_all[j] == atoms_list[i]:
                        atoms_position[i].append(list(
                            stru.get_scaled_positions()[j]))
                        atoms_masses[i] = stru.get_masses()[j]
                        atoms_magnetism[i] += np.linalg.norm(
                            stru.get_initial_magnetic_moments()[j])

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
        fd.write('ATOMIC_SPECIES\n')
        for i, elem in enumerate(atoms_list):
            pseudofile = pp[elem]
            temp1 = ' ' * (4 - len(atoms_list[i]))
            temp2 = ' ' * (14 - len(str(atoms_masses[i])))
            atomic_species = (atoms_list[i] + temp1
                              + str(atoms_masses[i]) + temp2
                              + pseudofile)

            fd.write(atomic_species)
            fd.write('\n')

        if basis is not None:
            fd.write('\n')
            fd.write('NUMERICAL_ORBITAL\n')
            for i, elem in enumerate(atoms_list):
                orbitalfile = basis[elem]
                fd.write(orbitalfile)
                fd.write('\n')

        if offsite_basis is not None:
            fd.write('\n')
            fd.write('ABFS_ORBITAL\n')
            for i, elem in enumerate(atoms_list):
                orbitalfile = offsite_basis[elem]
            fd.write(orbitalfile)
            fd.write('\n')

        fd.write('\n')
        fd.write('LATTICE_CONSTANT\n')
        fd.write(f'{1/Bohr} \n')
        fd.write('\n')

        fd.write('LATTICE_VECTORS\n')
        for i in range(3):
            for j in range(3):
                temp3 = str("{:0<12f}".format(
                    stru.get_cell()[i][j])) + ' ' * 3
                fd.write(temp3)
                fd.write('   ')
            fd.write('\n')
        fd.write('\n')

        fd.write('ATOMIC_POSITIONS\n')
        fd.write(coordinates_type)
        fd.write('\n')
        fd.write('\n')
        vel = stru.get_velocities()   # velocity in unit A/fs ?
        # mag = stru.get_magnetic_moments()
        for i in range(len(atoms_list)):
            fd.write(atoms_list[i])
            fd.write('\n')
            fd.write(str("{:0<12f}".format(float(atoms_magnetism[i]))))
            fd.write('\n')
            fd.write(str(len(atoms_position[i])))
            fd.write('\n')

            for j in range(len(atoms_position[i])):
                temp4 = str("{:0<12f}".format(
                    atoms_position[i][j][0])) + ' '
                temp5 = str("{:0<12f}".format(
                    atoms_position[i][j][1])) + ' '
                temp6 = str("{:0<12f}".format(
                    atoms_position[i][j][2])) + ' '
                sym_pos = temp4 + temp5 + temp6 + \
                    f'{fix[j][0]:.0f} {fix[j][1]:.0f} {fix[j][2]:.0f} '
                if init_vel:
                    sym_pos += f'v {vel[j][0]} {vel[j][1]} {vel[j][2]} '
                # if set_mag:
                #     if isinstance(mag[j], list):
                #         sym_pos += f'mag {mag[j][0]} {mag[j][1]} {mag[j][2]} '
                #     else:
                #         sym_pos += f'mag {mag[j]} '
                fd.write(sym_pos)
                fd.write('\n')
            fd.write('\n')


@writer
def write_abacus(fd,
                 atoms=None,
                 pp=None,
                 basis=None,
                 offsite_basis=None,
                 scaled=True,
                 init_vel=False):

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


@reader
def read_abacus(fd, latname=None, verbose=False):
    """Read structure information from abacus structure file.

    If `latname` is not None, 'LATTICE_VECTORS' should be removed in structure files of ABACUS. 
    Allowed values: 'sc', 'fcc', 'bcc', 'hexagonal', 'trigonal', 'st', 'bct', 'so', 'baco', 'fco', 'bco', 'sm', 'bacm', 'triclinic'

    If `verbose` is True, pseudo-potential and basis will be output along with the Atoms object.
    """

    from ase.constraints import FixCartesian

    contents = fd.read()
    title_str = r'(?:LATTICE_CONSTANT|NUMERICAL_ORBITAL|ABFS_ORBITAL|LATTICE_VECTORS|LATTICE_PARAMETERS|ATOMIC_POSITIONS)'

    # remove comments and empty lines
    contents = re.compile(r"#.*|//.*").sub('', contents)
    contents = re.compile(r'\n{2,}').sub('\n', contents)

    # specie, mass, pps
    specie_pattern = re.compile(
        rf'ATOMIC_SPECIES\s*\n([\s\S]+?)\s*\n{title_str}')
    specie_lines = np.array(
        [line.split() for line in specie_pattern.search(contents).group(1).split('\n')])
    symbols = specie_lines[:, 0]
    atom_mass = specie_lines[:, 1].astype(float)
    atom_potential = specie_lines[:, 2]
    ntype = len(symbols)

    # basis
    aim_title = 'NUMERICAL_ORBITAL'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    orb_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    orb_lines = orb_pattern.search(contents)
    if orb_lines:
        atom_basis = orb_lines.group(1).split('\n')
    else:
        atom_basis = []

    # ABFs basis
    aim_title = 'ABFS_ORBITAL'
    aim_title_sub = title_str.replace('|' + aim_title, '')
    abf_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    abf_lines = abf_pattern.search(contents)
    if abf_lines:
        atom_offsite_basis = abf_lines.group(1).split('\n')
    else:
        atom_offsite_basis = []

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
    except:  # colinear
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
        return atoms, atom_potential, atom_basis, atom_offsite_basis
    else:
        return atoms


def get_lattice_from_latname(lines, latname=None):

    from math import sqrt
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

    @lazymethod
    def _parse_cells(self):
        """Parse all the cells from the output file"""
        a0_pattern_str = rf'lattice constant \(Angstrom\)\s*=\s*({_re_float})'
        cell_pattern = re.compile(
            rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')
        alat = self.parse_scalar(a0_pattern_str)
        _lattice = np.reshape(cell_pattern.findall(
            self.contents), (-1, 3, 3)).astype(float)

        return _lattice * alat

    def get_cells(self, index):
        """Get cell from the output file according to index"""
        return self._parse_cells()[index]

    @lazyproperty
    def coordinate_system(self):
        """Parse coordinate system (Cartesian or Direct) from the output file"""
        # for '|', it will match all the patterns which results in '' or None
        class_pattern = re.compile(
            r'(DIRECT) COORDINATES|(CARTESIAN) COORDINATES')
        coord_class = list(class_pattern.search(self.contents).groups())
        remove_empty(coord_class)

        return coord_class[0]

    @lazymethod
    def _parse_site(self):
        """Parse sites for all the structures in the output file"""
        pos_pattern = re.compile(
            rf'(CARTESIAN COORDINATES \( UNIT = {_re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n')

        return pos_pattern.findall(self.contents)

    def get_site(self, index):
        """Get site information from the output file according to index"""
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
            remove_empty(data)
            site = list(map(list, site_pattern.findall(data[0])))
            list(map(remove_empty, site))
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

        return parse_block(self._parse_site()[index])

    @lazymethod
    def _parse_forces(self):
        """Parse all the forces from the output file"""
        force_pattern = re.compile(
            r'TOTAL\-FORCE\s*\(eV/Angstrom\)\n\n.*\s*atom\s*x\s*y\s*z\n([\s\S]+?)\n\n')

        return force_pattern.findall(self.contents)

    def get_forces(self, index):
        """Get forces from the output file according to index"""
        def str_to_force(val_in):
            data = []
            val = [v.strip().split() for v in val_in.split('\n')]
            for v in val:
                data.append(np.array(v[1:], dtype=float))
            return np.array(data)

        try:
            forces = self._parse_forces()[index]
            return str_to_force(forces)
        except:
            return

    @lazymethod
    def _parse_stress(self):
        """Parse the stress from the output file"""
        stress_pattern = re.compile(
            rf'(?:TOTAL\-|MD\s*)STRESS\s*\(KBAR\)\n\n.*\n\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')

        return stress_pattern.findall(self.contents)

    def get_stress(self, index):
        """Get the stress from the output file according to index"""
        from ase.stress import full_3x3_to_voigt_6_stress
        [index]

        try:
            stress = -0.1 * GPa * \
                np.array(self._parse_stress()[index]).reshape(
                    (3, 3)).astype(float)
            return full_3x3_to_voigt_6_stress(stress)
        except:
            return

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

    def get_eigenvalues(self, index):
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
            eigenvalues, occupations = [], []
            if nspin in [1, 4]:
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues.append(energies)
                occupations.append(occs)
            elif nspin == 2:
                val_up = re.search(
                    r'SPIN UP :([\s\S]+?)\n\nSPIN', val_in).group()
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_up)[:, 1, :]
                eigenvalues.append(energies)
                occupations.append(occs)

                val_dw = re.search(
                    r'SPIN DOWN :([\s\S]+?)(?:\n\n\s*EFERMI|\n\n\n)', val_in).group()
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_dw)[:, 1, :]
                eigenvalues.append(energies)
                occupations.append(occs)
            return np.array(eigenvalues), np.array(occupations)

        # NSCF
        def str_to_bandstructure(val_in):
            def extract_data(val):
                def func(i):
                    res = np.array(list(map(lambda x: x.strip().split(), re.search(
                        rf'k\-points{i+1}\(\d+\):.*\n([\s\S]+?)\n\n', val).group(1).split('\n'))))
                    return res[:, 2].astype(float), res[:, 3].astype(float)

                return np.asarray(list(map(func, [i for i in range(nks)])))

            nks = int(re.search(r'k\-points\d+\((\d+)\)', val_in).group(1))
            eigenvalues, occupations = [], []
            if re.search('spin up', val_in) and re.search('spin down', val_in):
                val = re.search(r'spin up :\n([\s\S]+?)\n\n\n', val_in).group()
                energies, occs = extract_data(
                    val)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues.append(energies[:int(nks / 2)])
                eigenvalues.append(energies[int(nks / 2):])
                occupations.append(occs[:int(nks / 2)])
                occupations.append(occs[int(nks / 2):])
            else:
                energies, occs = extract_data(
                    val_in)[:, 0, :], extract_data(val_in)[:, 1, :]
                eigenvalues.append(energies)
                occupations.append(occs)
            return np.array(eigenvalues), np.array(occupations)

        try:
            return str_to_energy_occupation(self._parse_eigenvalues()['scf'][index])
        except KeyError:
            return str_to_bandstructure(self._parse_eigenvalues()['nscf'][index])
        except:
            return

    @lazymethod
    def _parse_energy(self):
        """Parse the energy from the output file."""
        energy_pattern = re.compile(rf'\s*final etot is\s*({_re_float})\s*eV')

        return energy_pattern.findall(self.contents)

    def get_energy(self, index):
        """Get the energy from the output file according to index."""
        try:
            return float(self._parse_energy()[index])
        except:
            return


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
    def initial_cell(self):
        """The initial cell from the header of the running_*.log file"""
        return self.get_cells(0)

    @lazyproperty
    def initial_atoms(self):
        """Create an atoms object for the initial structure from the 
        header of the running_*.log file"""
        labels, positions, mag, vel = self.get_site(0)
        if self.coordinate_system == 'CARTESIAN':
            return Atoms(symbols=labels, positions=positions,
                         cell=self.initial_cell, pbc=True, velocities=vel, magmoms=mag)
        elif self.coordinate_system == 'DIRECT':
            return Atoms(symbols=labels, scaled_positions=positions,
                         cell=self.initial_cell, pbc=True, velocities=vel, magmoms=mag)

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

        return k_points, k_point_weights

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
        }


class AbacusOutCalcChunk(AbacusOutChunk):
    """A part of the running_*.log file correponding to a single structure"""

    def __init__(self, contents, header, index=-1):
        """Constructor

        Parameters
        ----------
        lines: str
            The contents of the output file
        header: dict
            A summary of the relevant information from the running_*.log header
        index: dict
            index of image
        """
        super().__init__(contents)
        self._header = header.header_summary
        self.index = index

    @lazyproperty
    def forces(self):
        """Parse the forces from the running_*.log file"""
        try:
            return self.get_forces(self.index)
        except:
            return

    @lazyproperty
    def stress(self):
        """Parse the stress from the running_*.log file"""
        try:
            return self.get_stress(self.index)
        except:
            return

    @lazyproperty
    def energy(self):
        """Parse the energy from the running_*.log file"""
        try:
            return self.get_energy(self.index)
        except:
            return

    @lazyproperty
    def eigenvalues(self):
        """All outputted eigenvalues for the system"""
        try:
            return self.get_eigenvalues(self.index)[0]
        except:
            return np.full(
                (self._header["n_spins"], self._header["n_k_points"], self._header["n_bands"]), np.nan)

    @lazyproperty
    def occupations(self):
        """All outputted occupations for the system"""
        try:
            return self.get_eigenvalues(self.index)[1]
        except:
            return np.full(
                (self._header["n_spins"], self._header["n_k_points"], self._header["n_bands"]), np.nan)

    @lazyproperty
    def kpts(self):
        """SinglePointKPoint objects"""
        return arrays_to_kpoints(self.eigenvalues, self.occupations, self._header["k_point_weights"])


@reader
def read_abacus_out(fd, index=-1):
    """Import ABACUS output files with all data available, i.e.
    relaxations, MD information, force information ..."""

    contents = fd.read()

    scaled_positions = None
    positions = None
    efermi = None
    energy = None
    forces = None
    stress = None
    ibzkpts = None
    kweights = None
    kpts = None
    images = []

    # cells
    # a0_pattern = re.compile(
    #     rf'lattice constant \(Bohr\)\s*=\s*({_re_float})')
    a0_pattern = re.compile(
        rf'lattice constant \(Angstrom\)\s*=\s*({_re_float})')
    # VB_pattern = re.compile(rf'Volume \(Bohr\^3\)\s*=\s*({_re_float})')
    # VB = float(VB_pattern.search(contents).group(1))
    # VA_pattern = re.compile(rf'Volume \(A\^3\)\s*=\s*({_re_float})')
    # VA = float(VA_pattern.search(contents).group(1))

    cell_pattern = re.compile(
        rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')
    # alat = float(a0_pattern.search(contents).group(1))*Bohr
    _lattice = np.reshape(cell_pattern.findall(
        contents)[-1], (3, 3)).astype(float)
    # alat = pow(VA/Cell(_lattice).volume, 1/3)
    alat = float(a0_pattern.search(contents).group(1))

    cell = Cell(_lattice * alat)

    # labels and positions
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

    pos_pattern = re.compile(
        rf'(CARTESIAN COORDINATES \( UNIT = {_re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n')
    site_pattern = re.compile(
        rf'tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})|tau[cd]_([a-zA-Z]+)\d+\s+({_re_float})\s+({_re_float})\s+({_re_float})\s+({_re_float})')
    class_pattern = re.compile(
        r'(DIRECT) COORDINATES|(CARTESIAN) COORDINATES')
    unit_pattern = re.compile(rf'UNIT = ({_re_float}) Bohr')
    # for '|', it will match all the patterns which results in '' or None
    coord_class = list(class_pattern.search(contents).groups())
    remove_empty(coord_class)
    coord_class = coord_class[0]
    data = list(pos_pattern.findall(contents)[-1])
    remove_empty(data)
    site = list(map(list, site_pattern.findall(data[0])))
    list(map(remove_empty, site))
    labels, pos, mag, vel = str_to_sites(site)
    if coord_class == 'CARTESIAN':
        unit = float(unit_pattern.search(contents).group(1)) * Bohr
        positions = pos * unit
    elif coord_class == 'DIRECT':
        scaled_positions = pos

    # kpoints
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
    sub_contents = k_pattern.search(contents).group(1)
    ibzkpts, kweights = str_to_kpoints(sub_contents)

    # parse eigenvalues and occupations
    def _parse_eig(res):
        _kpts = []
        for i, lspin in enumerate(res):
            for j, state in enumerate(res[lspin]):
                kpt = SinglePointKPoint(
                    kweights[j], i, ibzkpts[j], state.energies, state.occupations)
                _kpts.append(kpt)
        return _kpts

    # SCF: extract eigenvalues and occupations
    def str_to_energy_occupation(val_in):
        def extract_data(val_in, nks):
            State = namedtuple(
                'State', ['kpoint', 'energies', 'occupations', 'npws'])
            data = []
            for i in range(nks):
                kx, ky, kz, npws = re.search(
                    rf'{i+1}/{nks} kpoint \(Cartesian\)\s*=\s*({_re_float})\s*({_re_float})\s*({_re_float})\s*\((\d+)\s*pws\)', val_in).groups()
                res = np.array(list(map(lambda x: x.strip().split(), re.search(
                    rf'{i+1}/{nks} kpoint \(Cartesian\)\s*=.*\n([\s\S]+?)\n\n', val_in).group(1).split('\n'))), dtype=float)
                energies = res[:, 1]
                occupations = res[:, 2]
                state = State(kpoint=np.array([kx, ky, kz], dtype=float), energies=energies.astype(
                    float), occupations=occupations.astype(float), npws=int(npws))
                data.append(state)
            return data

        nspin = int(re.search(
            r'STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*(\d+)', val_in).group(1))
        nks = int(
            re.search(r'\d+/(\d+) kpoint \(Cartesian\)', val_in).group(1))
        data = dict()
        if nspin in [1, 4]:
            data['up'] = extract_data(val_in, nks)
        elif nspin == 2:
            val_up = re.search(
                r'SPIN UP :([\s\S]+?)\n\nSPIN', val_in).group()
            data['up'] = extract_data(val_up, nks)
            val_dw = re.search(
                r'SPIN DOWN :([\s\S]+?)(?:\n\n\s*EFERMI|\n\n\n)', val_in).group()
            data['down'] = extract_data(val_dw, nks)
        return data

    scf_eig_pattern = re.compile(
        r'(STATE ENERGY\(eV\) AND OCCUPATIONS\s*NSPIN\s*==\s*\d+[\s\S]+?(?:\n\n\s*EFERMI|\n\n\n))')
    scf_eig_all = scf_eig_pattern.findall(contents)
    if scf_eig_all:
        scf_eig = str_to_energy_occupation(scf_eig_all[-1])
        kpts = _parse_eig(scf_eig)

    # NSCF: extract eigenvalues and occupations
    def str_to_bandstructure(val_in):
        def extract_data(val_in, nks):
            State = namedtuple('State', ['kpoint', 'energies', 'occupations'])
            data = []
            for i in range(nks):
                kx, ky, kz = re.search(
                    rf'k\-points{i+1}\(\d+\):\s*({_re_float})\s*({_re_float})\s*({_re_float})', val_in).groups()
                res = np.array(list(map(lambda x: x.strip().split(), re.search(
                    rf'k\-points{i+1}\(\d+\):.*\n([\s\S]+?)\n\n', val_in).group(1).split('\n'))))
                energies = res[:, 2]
                occupations = res[:, 3]
                state = State(kpoint=np.array([kx, ky, kz], dtype=float), energies=energies.astype(
                    float), occupations=occupations.astype(float))
                data.append(state)
            return data

        nks = int(re.search(r'k\-points\d+\((\d+)\)', val_in).group(1))
        data = dict()
        if re.search('spin up', val_in) and re.search('spin down', val_in):
            val = re.search(r'spin up :\n([\s\S]+?)\n\n\n', val_in).group()
            val_new = extract_data(val, nks)
            data['up'] = val_new[:int(nks / 2)]
            data['down'] = val_new[int(nks / 2):]
        else:
            data['up'] = extract_data(val_in, nks)
        return data

    nscf_eig_pattern = re.compile(
        r'(band eigenvalue in this processor \(eV\)\s*:\n[\s\S]+?\n\n\n)')
    nscf_eig_all = nscf_eig_pattern.findall(contents)
    if nscf_eig_all:
        nscf_eig = str_to_bandstructure(nscf_eig_all[-1])
        kpts = _parse_eig(nscf_eig)

    # forces
    def str_to_force(val_in):
        data = []
        val = [v.strip().split() for v in val_in.split('\n')]
        for v in val:
            data.append(np.array(v[1:], dtype=float))
        return np.array(data)

    force_pattern = re.compile(
        r'TOTAL\-FORCE\s*\(eV/Angstrom\)\n\n.*\s*atom\s*x\s*y\s*z\n([\s\S]+?)\n\n')
    forces_all = force_pattern.findall(contents)
    if forces_all:
        forces = str_to_force(forces_all[-1])

    # stress
    stress_pattern = re.compile(
        rf'(?:TOTAL\-|MD\s*)STRESS\s*\(KBAR\)\n\n.*\n\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n\s*({_re_float})\s*({_re_float})\s*({_re_float})\n')
    stress_all = stress_pattern.findall(contents)
    if stress_all:
        stress = np.reshape(stress_all[-1], (3, 3)).astype(float)
        stress *= -0.1 * GPa
        stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

    # total/potential energy
    energy_pattern = re.compile(rf'\s*final etot is\s*({_re_float})\s*eV')
    energy_all = energy_pattern.findall(contents)
    if energy_all:
        energy = float(energy_all[-1])

    # fermi energy
    fermi_pattern = re.compile(rf'EFERMI\s*=\s*({_re_float})\s*eV')
    fermi_all = fermi_pattern.findall(contents)
    if fermi_all:
        efermi = float(fermi_all[-1])

    # extract energy(Ry), potential(Ry), kinetic(Ry), temperature(K) and pressure(KBAR) for MD
    md_pattern = re.compile(
        rf'Energy\s*Potential\s*Kinetic\s*Temperature\s*(?:Pressure \(KBAR\)\s*\n|\n)\s*({_re_float})\s*({_re_float})')
    md_all = md_pattern.findall(contents)

    # set atoms:
    if md_all:
        import os
        from glob2 import glob

        md_stru_file = os.path.join(
            os.path.dirname(fd.name), f'STRU_MD_*')
        md_stru_dir = os.path.join(
            os.path.dirname(fd.name), f'STRU')
        md_stru_dir_file = os.path.join(md_stru_dir, f'STRU_MD_*')
        if glob(md_stru_dir):
            files = glob(md_stru_dir)
        elif glob(md_stru_file):
            files = glob(md_stru_file)
        else:
            raise FileNotFoundError(
                f"Can't find {md_stru_file} or {md_stru_dir_file}")
        for i, file in enumerate(files):
            md_atoms = read_abacus(open(file, 'r'))
            md_e, md_pot = list(map(float, md_all[i]))
            md_atoms.calc = SinglePointDFTCalculator(md_atoms, energy=md_e * Hartree, free_energy=md_pot * Hartree,
                                                     forces=forces_all[i], stress=stress_all[i], efermi=fermi_all[i], ibzkpts=ibzkpts)
            md_atoms.calc.name = 'Abacus'
            images.append(md_atoms)
            # return requested images, code borrowed from ase/io/trajectory.py
        if isinstance(index, int):
            return images[index]
        else:
            step = index.step or 1
            if step > 0:
                start = index.start or 0
                if start < 0:
                    start += len(images)
                stop = index.stop or len(images)
                if stop < 0:
                    stop += len(images)
            else:
                if index.start is None:
                    start = len(images) - 1
                else:
                    start = index.start
                    if start < 0:
                        start += len(images)
                if index.stop is None:
                    stop = -1
                else:
                    stop = index.stop
                    if stop < 0:
                        stop += len(images)
            return [images[i] for i in range(start, stop, step)]
    else:
        if coord_class == 'CARTESIAN':
            atoms = Atoms(symbols=labels, positions=positions,
                          cell=cell, pbc=True, velocities=vel)
        elif coord_class == 'DIRECT':
            atoms = Atoms(symbols=labels, scaled_positions=scaled_positions,
                          cell=cell, pbc=True, velocities=vel)

        calc = SinglePointDFTCalculator(atoms, energy=energy, free_energy=energy,
                                        forces=forces, stress=stress, efermi=efermi, ibzkpts=ibzkpts)
        if kpts:
            calc.kpts = kpts
        calc.name = 'Abacus'
        atoms.calc = calc
        return [atoms]


def remove_empty(a: list):
    """Remove '' and [] in `a`"""
    while '' in a:
        a.remove('')
    while [] in a:
        a.remove([])
    while None in a:
        a.remove(None)


def handle_data(data):
    data.remove('')

    def handle_elem(elem):
        elist = elem.split(' ')
        remove_empty(elist)  # `list` will be modified in function
        return elist
    return list(map(handle_elem, data))
