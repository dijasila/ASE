from __future__ import annotations
import io
from typing import List, Union
import numpy as np

from ase import Atoms
from ase.utils import reader, writer
from ase.calculators.singlepoint import SinglePointCalculator


__author__ = "Hanyu Liu"
__email__ = "domainofbuaa@gmail.com"
__date__ = "2024-4-7"


@writer
def write_pwmat(fd: io.TextIOWrapper, atoms: Atoms) -> None:
    """
    Writes atom into PWmat input format
    
    Parameters
    ----------
    fd : TextIOWrapper
        File like object to which the atoms object should be written
    atoms : Atoms
        Input structure
    """
    fd.write(f"           {len(atoms)} atoms\n")
    fd.write("Lattice vector\n")
    for ii in range(3):
        fd.write(f"{atoms.cell[ii][0]:>15.10f} {atoms.cell[ii][1]:>15.10f} {atoms.cell[ii][2]:>15.10f}\n")
    fd.write("Position, move_x, move_y, move_z\n")
    rec_cell: np.ndarray = np.linalg.inv(np.array(atoms.get_cell()))
    frac_position: np.ndarray = np.dot(
        np.array(atoms.get_positions()),
        rec_cell)
    for ii in range(len(atoms)):
        fd.write(f"{atoms.get_atomic_numbers()[ii]:>5d} {frac_position[ii][0]:>15.10f} {frac_position[ii][1]:>15.10f} {frac_position[ii][2]:>15.10f}   1   1   1\n")
    fd.write("MAGNETIC\n")
    for ii in range(len(atoms)):
        fd.write(f"{atoms.get_atomic_numbers()[ii]:>5d} {atoms[ii].magmom:>10.5f}\n")


@reader
def read_pwmat(fd: io.TextIOWrapper) -> Atoms:
    """
    Read Atoms object from PWmat atom.config input/output file.

    Parameters
    ----------
    fd : io.TextIOWrapper
    
    Returns
    -------
    atoms : Atoms
        Atoms object
    """
    natoms: int = int(next(fd).split()[0])
    while ("LATTICE" in next(fd).upper()):
            break
    lattice: List[List[float]] = []
    positions: List[List[float]] = []
    numbers: List[int] = []
    magmoms: List[float] = []
    for _ in range(3):
        line = next(fd)
        lattice.append([float(val) for val in line.split()])
    while ("POSITION" in next(fd).upper()):
        break
    for _ in range(natoms):
        line_lst = next(fd).split()
        numbers.append(int(line_lst[0]))
        tmp_frac_position = np.array( [float(line_lst[1]), float(line_lst[2]), float(line_lst[3])] )
        tmp_cart_position = list(np.dot(tmp_frac_position, lattice).reshape(3))
        positions.append(tmp_cart_position)
    for ii in fd:
        if ("MAGNETIC" in ii):
            for _ in range(natoms):
                magmoms.append(float(next(fd).split()[1]))
        break

    return Atoms(cell=lattice, positions=positions, numbers=numbers, magmoms=magmoms)


@reader
def read_pwmat_report(fd: io.TextIOWrapper, index = -1):
    """Parse REPORT file

    Args:
        fd : io.TextIOWrapper
        
        index : Union[int, slice], optional]
            Defaults to -1.
    """
    def get_scf_block(fd: io.TextIOWrapper) -> List[str]:
        blk: List[str] = []
        for ii in fd:
            if not ii:
                return blk
            blk.append(ii)
            if "---------" in ii:
                return blk
        return blk

    def parse_scf_block(blk: List[str]):
        return float( blk[-2].split()[-2] )
    
    job: str = ""
    calculation: List[Atoms] = []
    for line in fd:
        if ("JOB" in line.upper()):
            job = line.split('=')[1].strip().upper()
            break
    if job == "SCF":
        while ("Weighted average num_of_PW for".upper() not in next(fd).upper()):
            pass
        for _ in range(5):
            next(fd)
        for line in fd:
            if "E_Fermi(eV)=" in line:      # End reading the blocks of NONSCF/SCF
                #efermi: float = float( line.split()[-1].strip() )
                break
            if line.split()[0] == "NONSCF":
                atoms = Atoms('H')   # 'H' represents NONSCF step.
                energy = 0.0
                atoms.calc = SinglePointCalculator(
                    atoms=atoms, energy=energy)
                calculation.append(atoms)
            elif line.split()[0] == "iter=":
                atoms = Atoms('He')   # 'H' represents SCF step.
                blk: List[str] = get_scf_block(fd)
                energy = parse_scf_block(blk)
                atoms.calc = SinglePointCalculator(
                    atoms=atoms, energy=energy)
                calculation.append(atoms)
    else:
        raise NotImplementedError(f"No explicit implementation for job task : {job}.")

    if calculation:
        if isinstance(index, int):
            steps = [calculation[index]]
        else:
            steps = calculation[index]
    else:
        steps = []

    for step in steps:
        yield step
    