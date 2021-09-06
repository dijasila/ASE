def test_safires():
    """Test for SAFIRES class.
    
    DOI: 10.1021/acs.jctc.1c00522
    
    SAFIRES is a boundary-based separation scheme
    for hybrid calculations. The fundamental features
    of the method can however be tested without doing
    an actual hybrid calculation (e.g. QM/MM)..

    The following unit test includes two tests; one for
    a simple, monoatomic argon LJ model and another one
    using H2O and the TIP4P force field.

    The LJ unit test is made redundant by the more
    complex H2O test. However, it is included to quickly
    rule out if any observed problem or instability is
    actually cause by SAFIRES or by changes to the 
    FixBondLengths or TIP4P classes.

    @bjk24, 2021-09-06.
    
    """

    import numpy as np
    from ase import units
    from ase import Atoms
    from ase.calculators.tip4p import TIP4P, rOH, angleHOH
    from ase.constraints import FixBondLengths
    from ase.constraints import FixAtoms
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet
    from ase.md.safires import SAFIRES

    # TEST 1: monoatomic LJ liquid

    # Build MWE LJ model system.
    a_cell = 30
    cell = ((a_cell, 0, 0), (0, a_cell, 0), (0, 0, a_cell))
    pbc = (1, 1, 1)

    atoms = Atoms("Ar3",
            [[(a_cell / 2), (a_cell / 2), (a_cell / 2)], 
            [(a_cell/2 + 2), (a_cell/2 - 1), (a_cell / 2)], 
            [(a_cell/2 - 4), (a_cell / 2 + 0.5), (a_cell / 2)]])
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)
    
    # Tags: 0 = solute, 1 = inner region, 2 = outer region.
    atoms.set_tags([0, 1, 2])

    # Fix solute particle (tag = 0)
    atoms.constraints = [FixAtoms(indices=[0])]

    # Apply initial momenta.
    atoms[1].momentum = np.asarray([1, 1, 0])
    atoms[2].momentum = np.asarray([2, 0, 0])
    
    # Initialize calculator and dynamics objects.
    atoms.calc = LennardJones(epsilon=120 * units.kB, 
                              sigma=3.4)
    dt = 1.0 * units.fs
    md = VelocityVerlet(atoms, timestep=dt)

    # Initialize SAFIRES class.
    safires = SAFIRES(atoms, mdobject=md, natoms=1, 
                      logfile=None)
    md.attach(safires.safires, interval=1)

    # Run MD.
    md.run(37)

    # Assert results.
    epot_check = -0.0204147939862
    epot = atoms.calc.results["energy"]
    d_check = 3.84968261723
    d = np.linalg.norm(atoms[0].position - atoms[1].position)
    
    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10

    # TEST 2: H2O / TIP4P

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
    atoms.set_tags([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    # Apply RATTLE constraint.
    rattle = rigid(atoms)
    fixatoms = FixAtoms(indices=[0, 1, 2])

    # Fix central molecule.
    atoms.constraints = ([fixatoms] + [rattle])
    atoms.set_positions(atoms.get_positions())

    # Apply initial momenta.
    atoms[3].momentum = np.asarray([-1, 1, 0])
    atoms[6].momentum = np.asarray([-1, -1, -1])

    # Initialize calculator and dynamics objects.
    atoms.calc = TIP4P(rc=14.9)
    dt = 1.0 * units.fs
    md = VelocityVerlet(atoms, timestep=dt)

    # Initialize SAFIRES class.
    safires = SAFIRES(atoms, mdobject=md, natoms=3, 
                      logfile=None)
    md.attach(safires.safires, interval=1)
    
    # Run MD.
    md.run(28)
    
    # Assert results.
    epot_check = -0.083096559019
    epot = atoms.calc.results["energy"]
    d_check = 2.70871422216
    d = np.linalg.norm(atoms[0].position - atoms[3].position)
    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10
