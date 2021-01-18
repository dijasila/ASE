def test_safires():
    
    import numpy as np
    from ase import units
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.md.verlet import VelocityVerlet
    from ase.md.safires import SAFIRES
    from ase.constraints import FixAtoms

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
    fixatoms = FixAtoms(indices=[atom.index for atom in atoms if atom.tag == 0])
    atoms.constraints = ([fixatoms])

    atoms.set_calculator(LennardJones(epsilon=120 * units.kB,
                                      sigma=3.4))
    atoms[1].momentum = np.asarray([1, 1, 0])
    atoms[2].momentum = np.asarray([2, 0, 0])

    dt = 1.0 * units.fs
    md = VelocityVerlet(atoms, timestep=dt)

    safires = SAFIRES(atoms, mdobject=md, natoms=1,
                      logfile="md.log")
    md.attach(safires.safires, interval=1)

    md.run(37)

    epot_check = -0.0172312389809
    epot = atoms.calc.results["energy"]
    d_check = 3.85006966993
    d = np.linalg.norm(atoms[0].position - atoms[1].position)

    assert abs(epot_check - epot) < 1.e-10
    assert abs(d_check - d) < 1.e-10 
