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

from ase import Atoms
from ase.utils import reader, writer
from ase.units import Bohr, Hartree, GPa
re_float = r'[\d\.\-\+Ee]+'

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
                          set_vel=False,
                          set_mag=False):
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
            temp1 = ' ' * (4-len(atoms_list[i]))
            temp2 = ' ' * (14-len(str(atoms_masses[i])))
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
        fd.write('1.889726125 \n')
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
        mag = stru.get_initial_magnetic_moments()
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
                if set_vel:
                    sym_pos += f'v {vel[j][0]} {vel[j][1]} {vel[j][2]} '
                if set_mag:
                    if isinstance(mag[j], list):
                        sym_pos += f'mag {mag[j][0]} {mag[j][1]} {mag[j][2]} '
                    else:
                        sym_pos += f'mag {mag[j]} '
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
                 set_vel=False,
                 set_mag=False):

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
                    fix_cart[constr.a] = constr.mask

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
                              set_vel,
                              set_mag)


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
    specie_pattern = re.compile(rf'ATOMIC_SPECIES\s*\n([\s\S]+?)\s*\n{title_str}')
    specie_lines = np.array([line.split() for line in specie_pattern.search(contents).group(1).split('\n')])
    symbols = specie_lines[:, 0]
    atom_mass = specie_lines[:, 1].astype(float)
    atom_potential = specie_lines[:, 2]
    ntype = len(symbols)

    # basis
    aim_title = 'NUMERICAL_ORBITAL'
    aim_title_sub = title_str.replace('|'+aim_title, '')
    orb_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    orb_lines = orb_pattern.search(contents)
    if orb_lines:
        atom_basis = orb_lines.group(1).split('\n')
    else:
        atom_basis = []

    # ABFs basis
    aim_title = 'ABFS_ORBITAL'
    aim_title_sub = title_str.replace('|'+aim_title, '')
    abf_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    abf_lines = abf_pattern.search(contents)
    if abf_lines:
        atom_offsite_basis = abf_lines.group(1).split('\n')
    else:
        atom_offsite_basis = []

    # lattice constant
    aim_title = 'LATTICE_CONSTANT'
    aim_title_sub = title_str.replace('|'+aim_title, '')
    a0_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
    a0_lines = a0_pattern.search(contents)
    atom_lattice_scale = float(a0_lines.group(1))

    # lattice vector
    if latname:
        aim_title = 'LATTICE_PARAMETERS'
        aim_title_sub = title_str.replace('|'+aim_title, '')
        lparam_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
        lparam_lines = lparam_pattern.search(contents)
        atom_lattice = get_lattice_from_latname(lparam_lines, latname)
    else:
        aim_title = 'LATTICE_VECTORS'
        aim_title_sub = title_str.replace('|'+aim_title, '')
        vec_pattern = re.compile(rf'{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}')
        vec_lines = vec_pattern.search(contents)
        if vec_lines:
            atom_lattice = np.array([line.split() for line in vec_pattern.search(contents).group(1).split('\n')]).astype(float)
        else:
            raise Exception(f"Parameter `latname` or `LATTICE_VECTORS` in {fd.name} must be set.")
    atom_lattice = atom_lattice * atom_lattice_scale * Bohr

    aim_title = 'ATOMIC_POSITIONS'
    type_pattern = re.compile(rf'{aim_title}\s*\n(\w+)\s*\n')
    # type of coordinates
    atom_pos_type = type_pattern.search(contents).group(1)
    assert atom_pos_type in ['Direct', 'Cartesian'], "Only two type of atomic coordinates are supported: 'Direct' or 'Cartesian'."

    block_pattern = re.compile(rf'{atom_pos_type}\s*\n([\s\S]+)')
    block = block_pattern.search(contents).group()
    atom_magnetism = []
    atom_symbol = []
    atom_block = []
    for i, symbol in enumerate(symbols):
        pattern = re.compile(rf'{symbol}\s*\n({re_float})\s*\n(\d+)')
        sub_block = pattern.search(block)
        number = int(sub_block.group(2))

        # symbols, magnetism
        sym = [symbol]*number
        atom_mags = [float(sub_block.group(1))]*number
        for j in range(number):
            atom_symbol.append(sym[j])
            atom_magnetism.append(atom_mags[j])

        if i == ntype-1:
            lines_pattern = re.compile(rf'{symbol}\s*\n{re_float}\s*\n\d+\s*\n([\s\S]+)\s*\n')
        else:
            lines_pattern = re.compile(rf'{symbol}\s*\n{re_float}\s*\n\d+\s*\n([\s\S]+?)\s*\n\w+\s*\n{re_float}')
        lines = lines_pattern.search(block)
        for j in [line.split() for line in lines.group(1).split('\n')]:
            atom_block.append(j)
    atom_block = np.array(atom_block)
    atom_magnetism = np.array(atom_magnetism)

    # position
    atom_positions = atom_block[:, 0:3].astype(float)
    natoms = len(atom_positions)

    # fix_cart 
    if (atom_block[:, 3] == ['m']*natoms).all():
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
            res = atom_block[:, index+1:index+1+num].astype(float)

        return res, index
    
    # velocity
    v_labels = ['v', 'vel', 'velocity']
    atom_vel, v_index = _get_index(v_labels, 3)
    
    # magnetism
    m_labels = ['mag', 'magmom']
    if 'angle1' in atom_block or 'angle2' in atom_block:
        warnings.warn("Non-colinear angle-settings are not yet supported for this interface.")
    mags, m_index = _get_index(m_labels, 1)
    try:     # non-colinear
        if m_index:
            atom_magnetism = atom_block[:, m_index+1:m_index+4].astype(float)
    except:  # colinear
        if m_index:
            atom_magnetism = mags

    # to ase
    if atom_pos_type == 'Direct':
        atoms = Atoms(symbols=atom_symbol,
                          cell=atom_lattice,
                          scaled_positions=atom_positions,
                          pbc=True)
    elif atom_pos_type== 'Cartesian':
        atoms = Atoms(symbols=atom_symbol,
                          cell=atom_lattice,
                          positions=atom_positions*atom_lattice_scale*Bohr,
                          pbc=True)   
    if v_index:
        atoms.set_velocities(atom_vel)
    if atom_magnetism.any():
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
        return np.array([[1.0, 0, 0], [-0.5, sqrt(3)/2, 0], [0, 0, x]])
    elif latname == 'trigonal':
        x = float(lines[0])
        tx=sqrt((1-x)/2)
        ty=sqrt((1-x)/6)
        tz=sqrt((1+2*x)/3)
        return np.array([[tx, -ty, tz], [0, 2*ty, tz], [-tx, -ty, tz]])
    elif latname == 'st':
        x = float(lines[0])
        return np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, x]])
    elif latname == 'bct':
        x = float(lines[0])
        return np.array([[0.5, -0.5, x], [0.5, 0.5, x], [0.5, 0.5, x]])
    elif latname == 'baco':
        x, y = list(map(float, lines))
        return np.array([[0.5, x/2, 0], [-0.5, x/2, 0], [0, 0, y]])
    elif latname == 'fco':
        x, y = list(map(float, lines))
        return np.array([[0.5, 0, y/2], [0.5, x/2, 0], [0.5, x/2, 0]])
    elif latname == 'bco':
        x, y = list(map(float, lines))
        return np.array([[0.5, x/2, y/2], [-0.5, x/2, y/2], [-0.5, -x/2, y/2]])
    elif latname == 'bco':
        x, y, z = list(map(float, lines))
        return np.array([[1, 0, 0], [x*z, x*sqrt(1-z**2), 0], [0, 0, y]])
    elif latname == 'bacm':
        x, y, z = list(map(float, lines))
        return np.array([[0.5, 0, -y/2], [x*z, x*sqrt(1-z**2), 0], [0.5, 0, y/2]])
    elif latname == 'triclinic':
        x, y, m, n, l = list(map(float, lines))
        fac = sqrt(1+2*m*n*l-m**2-n**2-l**2)/sqrt(1-m**2)
        return np.array([[1, 0, 0], [x*m, x*sqrt(1-m**2), 0], [y*n, y*(l-n*m/sqrt(1-m**2)), y*fac]])


@reader
def read_abacus_out(fd, index=-1):
    """Import ABACUS output files with all data available, i.e.
    relaxations, MD information, force information ..."""
    import re
    import os
    from ase import Atom
    from ase.cell import Cell
    from ase.data import atomic_masses, atomic_numbers
    from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint

    def _set_eig_occ():
        eigs, occs = [], []
        eigsfull, occsfull = [], []
        ik, ib = 0, 0
        for i in range(totline):
            if i % (nbands+2) == 0:
                ik += 1
                next(fd)
            elif (i+1) % (nbands+2) == 0:
                ik = 0
                next(fd)
                continue
            else:
                sline = next(fd).split()
                if len(sline) == 3:
                    eig, occ = sline[1:]
                elif len(sline) == 4:
                    eig, occ = sline[2:]
                eigs.append(float(eig))
                occs.append(float(occ))
                ib += 1
                if ib == nbands:
                    eigsfull.append(eigs)
                    occsfull.append(occs)
                    ib = 0
                    eigs, occs = [], []
        return eigsfull, occsfull

    molecular_dynamics = False
    images = []
    scaled_position = None
    position = None
    efermi = None
    energy = None
    force = None
    stress = None
    nbands = None
    natom = None
    nspin = None
    nkstot = None
    nkstot_ibz = None
    ibzkpts = None
    kpts = None
    eigenvalues, occupations = [], []

    # find flag
    k_find = False

    for line in fd:
        if 'Version' in line:
            ver = 'ABACUS version: ' + ' '.join(line.split()[1:])

        # extract total numbers
        if "TOTAL ATOM NUMBER" in line:
            natom = int(line.split()[-1])

        # extract position
        if 'DIRECT COORDINATES' in line and not molecular_dynamics:
            scaled_position = []
            next(fd)
            atoms = Atoms()
            for i in range(natom):
                at, x, y, z, mag, vx, vy, vz = next(
                    fd).split()  # velocity in unit A/fs ?
                at = re.match('[a-zA-Z]+', at.split('_')[-1]).group()
                scaled_position.append(list(map(float, [x, y, z])))
                momentum = atomic_masses[atomic_numbers[at]
                                         ]*np.array([vx, vy, vz], dtype=float)
                atoms.append(Atom(symbol=at, momentum=momentum, magmom=mag))
        if 'CARTESIAN COORDINATES' in line and not molecular_dynamics:
            position = []
            next(fd)
            atoms = Atoms()
            for i in range(natom):
                at, x, y, z, mag, vx, vy, vz = next(
                    fd).split()  # velocity in unit A/fs ?
                at = re.match('[a-zA-Z]+', at.split('_')[-1]).group()
                position.append(list(map(float, [x, y, z])))
                momentum = atomic_masses[atomic_numbers[at]
                                         ]*np.array([vx, vy, vz], dtype=float)
                atoms.append(Atom(symbol=at, momentum=momentum, magmom=mag))

        # extract cell
        if "Volume (Bohr^3)" in line and not molecular_dynamics:
            V_b = float(line.split()[-1])
        if "Volume (A^3)" in line and not molecular_dynamics:
            V_a = float(line.split()[-1])
        if "Lattice vectors: (Cartesian coordinate: in unit of a_0)" in line and not molecular_dynamics:
            cell = []
            a_0 = pow(V_b/V_a, 1/3)*Bohr
            for i in range(3):
                ax, ay, az = next(fd).split()
                cell.append([float(ax)*a_0, float(ay)*a_0, float(az)*a_0])
            cell = Cell(cell)
            if scaled_position:
                position = cell.cartesian_positions(scaled_position)
            else:
                position = np.array(position)*a_0

        # extract nbands, nspin and nkstot/nkstot_ibz
        if 'NBANDS' in line:
            nbands = int(line.split()[-1])

        # extract md information
        if 'STEP OF MOLECULAR DYNAMICS' in line:
            molecular_dynamics = True
            md_step = int(line.split()[-1])
            md_stru_file = os.path.join(
                os.path.dirname(fd.name), f'STRU_MD_{md_step}')
            md_stru_dir = os.path.join(
                os.path.dirname(fd.name), f'STRU')
            md_stru_dir_file = os.path.join(md_stru_dir, f'STRU_MD_{md_step}')
            if os.path.exists(md_stru_file):
                md_atoms = read_abacus(open(md_stru_file, 'r'))
                images.append(md_atoms)
            elif os.path.exists(md_stru_dir_file):   # compatible with ABACUS v2.2.3
                md_atoms = read_abacus(open(md_stru_dir_file, 'r'))
                images.append(md_atoms)
            else:
                raise FileNotFoundError(f"Can't find {md_stru_file} or {md_stru_dir_file}")

        # extract ibzkpts
        if 'SETUP K-POINTS' in line and not k_find:
            nspin = int(next(fd).split()[-1])
            next(fd)
            sline = next(fd).split()
            if sline[0] == 'nkstot':
                nkstot = int(sline[-1])
            else:
                nkstot = int(next(fd).split()[-1])
            sline = next(fd).split()
            if sline and sline[0] == 'nkstot_ibz':
                nkstot_ibz = int(sline[-1])
            next(fd)
            nks = nkstot_ibz if nkstot_ibz else nkstot
            totline = (nbands+2)*nks
            ibzkpts = []
            weights = []
            for i in range(nks):
                kindex, kx, ky, kz, wei = next(fd).split()[:5]
                ibzkpts.append(list(map(float, [kx, ky, kz])))
                weights.append(float(wei))
            ibzkpts = np.array(ibzkpts)
            weights = np.array(weights)
            k_find = True

        # extract bands and occupations
        # ABACUS nscf.log without efermi
        if ('STATE ENERGY(eV) AND OCCUPATIONS    NSPIN == 1' in line) or ('band eigenvalue in this processor (eV)' in line and nspin == 1):
            eigs, occs = _set_eig_occ()
            eigenvalues.append(eigs)
            occupations.append(occs)
            if eigenvalues and occupations:
                kpts = []
                for w, k, e, o in zip(weights, ibzkpts, eigenvalues[0], occupations[0]):
                    kpt = SinglePointKPoint(w, 0, k, e, o)
                    kpts.append(kpt)

        if ('STATE ENERGY(eV) AND OCCUPATIONS    NSPIN == 2' in line) or ('band eigenvalue in this processor (eV)' in line and nspin == 2):
            eigenvalues, occupations = [], []
            sline = next(fd)
            if 'SPIN UP' in sline or 'spin up' in sline:
                eigs, occs = _set_eig_occ()
                eigenvalues.append(eigs)
                occupations.append(occs)
            sline = next(fd)
            if 'SPIN DOWN' in sline or 'spin down' in sline:
                eigs, occs = _set_eig_occ()
                eigenvalues.append(eigs)
                occupations.append(occs)
            if eigenvalues and occupations:
                kpts = []
                for s in range(nspin):
                    for w, k, e, o in zip(weights, ibzkpts, eigenvalues[s], occupations[s]):
                        kpt = SinglePointKPoint(w, s, k, e, o)
                        kpts.append(kpt)

        # extract force
        if "TOTAL-FORCE (eV/Angstrom)" in line:
            force = np.zeros((natom, 3))
            for i in range(4):
                next(fd)
            for i in range(natom):
                element, fx, fy, fz = next(fd).split()
                force[i] = [float(fx), float(fy), float(fz)]

        # extract energy(Ry), potential(Ry), kinetic(Ry), temperature(K) and pressure(KBAR) for MD
        if "Energy              Potential           Kinetic             Temperature         Pressure (KBAR)" in line:
            md_e, md_pot = map(float, next(fd).split())[:2]
            md_e *= Hartree
            md_pot *= Hartree

        # extract stress
        if "TOTAL-STRESS (KBAR)" in line:
            stress = np.zeros((3, 3))
            for i in range(3):
                next(fd)
            for i in range(3):
                sx, sy, sz = next(fd).split()
                stress[i] = [float(sx), float(sy), float(sz)]
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]
        elif "MD STRESS (KBAR)" in line:
            stress = np.zeros((3, 3))
            for i in range(3):
                next(fd)
            for i in range(3):
                sx, sy, sz = next(fd).split()
                stress[i] = [float(sx), float(sy), float(sz)]
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]
            # if nkstot_ibz:
            images[-1].calc = SinglePointDFTCalculator(md_atoms, energy=md_e, free_energy=md_pot,
                                                       forces=force, stress=stress, efermi=efermi, ibzkpts=ibzkpts)
            # elif nkstot:
            #     images[-1].calc = SinglePointDFTCalculator(atoms, energy=energy,
            #                             forces=force, stress=stress, efermi=efermi, bzkpts=ibzkpts)
            images[-1].calc.name = 'Abacus'

        # extract efermi
        if "E_Fermi" in line:
            efermi = float(line.split()[-1])

        # extract energy
        if "final etot is" in line:
            energy = float(line.split()[-2])

    fd.close()
    if not molecular_dynamics:
        atoms.set_cell(cell)
        atoms.set_positions(position)
        # if nkstot_ibz:
        calc = SinglePointDFTCalculator(atoms, energy=energy, free_energy=energy,
                                        forces=force, stress=stress, efermi=efermi, ibzkpts=ibzkpts)
        # elif nkstot:
        #     calc = SinglePointDFTCalculator(atoms, energy=energy,
        #                                 forces=force, stress=stress, efermi=efermi, bzkpts=ibzkpts)

        if kpts:
            calc.kpts = kpts
        calc.name = 'Abacus'
        atoms.calc = calc
        
        return [atoms]
    else:
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
