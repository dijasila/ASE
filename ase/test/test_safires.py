def test_safires():
    """Tests the SAFIRES boundary method.

    Creates a system of three Lennard-Jones particles
    with Ar parameters, one being the solute (tag = 0),
    one representing the inner region (tag = 1), and
    one the outer region (tag = 2). A boundary event
    occurs on step 36, so the potential energy and
    distance between solute and inner region particle
    at step 37 can be used to assess energy conservation
    (NVE dynamics) and if the elastic collision is
    performed as expected.
    """

    import numpy as np
    from ase import units
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet
    from ase.md.safires import SAFIRES
    from ase.constraints import FixAtoms

    # Build MWE model system.
    a_cell = 10
    cell = ((a_cell, 0, 0), (0, a_cell, 0), (0, 0, a_cell))
    pbc = (1, 1, 1)

    atoms = Atoms("Ar3",
                  [[(a_cell / 2), (a_cell / 2), (a_cell / 2)],
                  [(a_cell/2 + 2), (a_cell/2 - 1), (a_cell / 2)],
                  [(a_cell/2 - 4), (a_cell / 2 + 0.5), (a_cell / 2)]])
    atoms.set_tags([0,1,2]) # 0: solute, 1: inner region, 2: outer region
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)
    atoms.calc = LennardJones(epsilon=120 * units.kB, sigma=3.4)

    # Fix constrain solute.
    fixatoms = FixAtoms(indices=[atom.index for atom in atoms
                                 if atom.tag == 0])
    atoms.constraints = ([fixatoms])

    # Give atoms initial momentum.
    atoms[1].momentum = np.asarray([1, 1, 0])
    atoms[2].momentum = np.asarray([2, 0, 0])

    # Initialize Velocity Verlet NVE dynamics.
    dt = 1.0 * units.fs
    md = VelocityVerlet(atoms, timestep=dt)

    # Initialize SAFIRES class.
    safires = SAFIRES(atoms, mdobject=md, natoms=1,
                      logfile="md.log")
    md.attach(safires.safires, interval=1)

    # Run.
    md.run(37)

    # Evaluate potential energy and solute -> tag=1 distance.
    epot_check = -0.0172312389809
    epot = atoms.calc.results["energy"]
    d_check = 3.85006966993
    d = np.linalg.norm(atoms[0].position - atoms[1].position)
    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10
