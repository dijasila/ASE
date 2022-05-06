def test_safires():
    """Test for SAFIRES class.
    
    SAFIRES is a boundary-based separation scheme
    for hybrid calculations. The fundamental features
    of the method can however be tested without doing
    an actual hybrid calculation (e.g. QM/MM).

    The first test checks if SAFIRES can successfully
    resolve a boundary event where 2 collisions between
    three LJ particles occur during the same time step,
    which constitutes a tricky fringe case.

    The second test checks if a collision between two
    water molecules can be resolved successfully, which
    tests the interplay with FixBondLengths constraints
    and the internal COM reduction scheme used to treat
    molecules.

    For more information, check the original publication.
    DOI: 10.1021/acs.jctc.1c00522
    """
    import numpy as np
    from ase import units
    from ase import Atoms
    from ase.calculators.tip4p import TIP4P, rOH, angleHOH
    from ase.calculators.lj import LennardJones
    from ase.constraints import FixBondLengths
    from ase.constraints import FixAtoms
    from ase.md.safires import SAFIRES

    # TEST 1: double collision between 3 LJ particles.
    
    # Build MWE LJ model system.
    a_cell = 30
    cell = ((a_cell, 0, 0), (0, a_cell, 0), (0, 0, a_cell))
    pbc = (1, 1, 1)

    atoms = Atoms("Ar4",
            [[(a_cell / 2), (a_cell / 2), (a_cell / 2)],
            [(a_cell / 2 + 3.9), (a_cell / 2), (a_cell / 2)],
            [(a_cell / 2), (a_cell / 2 + 4), (a_cell / 2)],
            [(a_cell / 2 - 4.1), (a_cell / 2), (a_cell / 2)]])
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)

    # Tags: 0 = solute, 1 = inner region, 2 = outer region.
    atoms.set_tags([1, 2, 2, 3])

    # Fix solute particle (tag = 0)
    atoms.constraints = [FixAtoms(indices=[0])]

    # Apply initial momenta.
    atoms[1].momentum = np.asarray([5.4, 0, 0])
    atoms[2].momentum = np.asarray([0, 2, 0])
    atoms[3].momentum = np.asarray([2, 0, 0])

    # Initialize calculator and dynamics objects.
    atoms.calc = LennardJones(epsilon=120 * units.kB,
                              sigma=3.4)
    dt = 2.0 * units.fs
    md = SAFIRES(atoms, timestep=dt, natoms=1, friction=0,
                 temperature_K=0, logfile="1atoms.log",
                 fixcm=True)

    # Run MD.
    md.run(6)

    # Assert results.
    epot_check = -0.03151762881913554
    epot = atoms.calc.results["energy"]
    d_check = 4.058863010909957
    d = np.linalg.norm(atoms[0].position - atoms[1].position)

    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10

    # TEST 2: H2O / TIP4P-RATTLE test.

    # Reset atoms object.
    atoms = Atoms()

    # RATTLE constraint to fix bond lengths of H2O molecules.
    def rigid(atoms):
        rattle = ([(3 * i + j, 3 * i + (j + 1) % 3)
                            for i in range(len(atoms) // 3)
                            for j in [0, 1, 2]])

        rattle = [(c[0], c[1]) for c in rattle]
        rattle = FixBondLengths(rattle)
        return rattle

    # Build H2O MWE model systen.
    a_cell = 30
    cell = ((a_cell, 0, 0), (0, a_cell, 0), (0, 0, a_cell))
    pbc = (1, 1, 1)

    x = angleHOH * np.pi / 180 / 2
    pos = [[5, 5, 5], 
           [5, 5 + rOH*np.cos(x), 5 + rOH*np.sin(x)], 
           [5, 5 + rOH*np.cos(x), 5 - rOH*np.sin(x)], 
           [3.5, 3.5, 3.5], 
           [3.5, 3.5 + rOH*np.cos(x), 3.5 + rOH*np.sin(x)], 
           [3.5, 3.5 + rOH*np.cos(x), 3.5 - rOH*np.sin(x)], 
           [6.7, 6.7, 6.7], 
           [6.7, 6.7 + rOH*np.cos(x), 6.7 + rOH*np.sin(x)], 
           [6.7, 6.7 + rOH*np.cos(x), 6.7 - rOH*np.sin(x)]]
    atoms = Atoms('OH2OH2OH2', positions=pos, cell=cell, pbc=pbc)
    atoms.center()

    # Tags: 0 = solute, 1 = inner region, 2 = outer region.
    atoms.set_tags([1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    # Apply RATTLE constraint.
    rattle = rigid(atoms)
    fixatoms = FixAtoms(indices=[0, 1, 2])

    # Fix central molecule.
    atoms.constraints = ([fixatoms] + [rattle])
    atoms.set_positions(atoms.get_positions())

    # Apply initial momenta.
    atoms[3].momentum = np.asarray([-10, 10, 0])
    atoms[6].momentum = np.asarray([-10, -10, -10])

    # Initialize calculator and dynamics objects.
    atoms.calc = TIP4P(rc=14.9)
    dt = 1.0 * units.fs
    md = SAFIRES(atoms, timestep=dt, natoms=3, temperature_K=0,
                 friction=0, logfile="3atoms.log", fixcm=True)

    # Run MD.
    md.run(4)
    
    # Assert results.
    epot_check = 0.0280818883214471
    epot = atoms.calc.results["energy"]
    d_check = 2.583935087982253
    d = np.linalg.norm(atoms[0].position - atoms[3].position)
    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10
