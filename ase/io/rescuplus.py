"""Reads RESCU+ files.

Read multiple structures and results from ``rescuplus_scf`` output files. Read
structures from ``rescuplus_scf`` input files.

"""
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.singlepoint import SinglePointKPoint
from rescupy import TotalEnergy


def read_rescu_out(fileobj):
    """Reads RESCU+ output files.

    The atomistic configurations as well as results (energy, force, stress,
    magnetic moments) of the calculation are read for all configurations
    within the output JSON file.

    Parameters
    ----------
    fileobj : file|str
        A file object or filename

    Return
    ------
    structure : Atoms
        The Atoms has a SinglePointCalculator attached with any results
        parsed from the file.

    """
    if isinstance(fileobj, str):
        fileobj = open(fileobj, 'rU')

    ecalc = TotalEnergy.read(fileobj, units='si')
    cell = ecalc.system.cell.avec
    symbols = ecalc.system.atoms.get_symbols(standard=True)
    positions = ecalc.system.atoms.get_positions(ecalc.system.cell)
    constraint = None
    structure = Atoms(symbols=symbols, positions=positions, cell=cell,
                      constraint=constraint, pbc=True)

    # Extract calculation results
    energy = ecalc.energy.etot
    efermi = ecalc.energy.efermi
    forces = ecalc.energy.forces
    if not ecalc.energy.forces_return:
        forces = None
    stress = -ecalc.energy.stress
    if not ecalc.energy.stress_return:
        stress = None
    magnetic_moments = None

    # K-points
    ibzkpts = ecalc.system.kpoint.fractional_coordinates
    weights = ecalc.system.kpoint.kwght

    # Bands
    nkpt = ibzkpts.shape[0]
    nspin = 1
    eigenvalues = ecalc.energy.eigenvalues
    eigenvalues = eigenvalues.reshape((nkpt, -1, nspin))    # [kpt,band,spin]

    kpts = []
    for s in range(nspin):
        for w, k, e in zip(weights, ibzkpts, eigenvalues[:, :, s]):
            kpt = SinglePointKPoint(w, s, k, eps_n=e)
            kpts.append(kpt)

    # Put everything together
    calc = SinglePointDFTCalculator(structure, energy=energy, forces=forces,
                                    stress=stress, efermi=efermi,
                                    magmoms=magnetic_moments, ibzkpts=ibzkpts,
                                    bzkpts=ibzkpts)
    calc.kpts = kpts
    structure.calc = calc

    return structure, ecalc


def write_rescu_in(fd, atoms, input_data={}, pseudopotentials=None,
                   kpts=None, gamma_centered=None, **kwargs):
    """
    Create an input file for ``rescuplus_scf``.

    Units are automatically converted from (ang/ev to bohr/hartree).

    Parameters
    ----------
    fd: file
        A file like object to write the input file to.
    atoms: Atoms
        A single atomistic configuration to write to `fd`.
    input_data: dict
        A nested dictionary with input parameters for ``rescuplus_scf``.
    pseudopotentials: list
        A list of dictionaries, one for each atomic species, e.g.
        [{'label':'Ga', 'path':'Ga_AtomicData.mat'},
           {'label':'As', 'path':'As_AtomicData.mat'}].
    kpts: array
        List of 3 integers giving the dimensions of a Monkhorst-Pack grid.
        If ``kpts`` is set to ``None``, only the Γ-point will be included.
    """

    # init input_data dict
    if 'atoms' not in input_data['system'].keys():
        input_data['system']['atoms'] = {}
    if 'cell' not in input_data['system'].keys():
        input_data['system']['cell'] = {}
    if 'kpoint' not in input_data['system'].keys():
        input_data['system']['kpoint'] = {}
    # atoms
    input_data['system']['atoms']['positions'] = atoms.positions
    # PP
    if pseudopotentials is not None:
        input_data['system']['atoms']['species'] = pseudopotentials
    if 'formula' not in input_data['system']['atoms'].keys():
        input_data['system']['atoms']['formula'] = "".join(atoms.get_chemical_symbols())
    # kpoints - MP grid
    if kpts is not None:
        input_data['system']['kpoint']['grid'] = kpts
    if gamma_centered is not None:
        input_data['system']['kpoint']['gamma_centered'] = gamma_centered
    # cell
    input_data['system']['cell']['avec'] = atoms.cell
    # write file
    ecalc = TotalEnergy(**input_data)
    ecalc.system.atoms.formula = ecalc.system.atoms.get_formula(format="short")
    ecalc.write(fd, units="atomic")
