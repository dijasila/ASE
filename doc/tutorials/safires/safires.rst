.. _safires:

===============================
SAFIRES for coupled simulations
===============================

SAFIRES is a boundary method used to separate a model system into
two regions to be calculated using different computational
methodologies. This tutorial aims to introduce basic usage of
SAFIRES using the simple Lennard-Jones model potential
(:class:`~ase.calculators.lj.LennardJones`), *i.e.* using the same
methodology for the inner and outer region.

General requirements
--------------------

In order to use SAFIRES with a model system, certain requirements
need to be fulfilled.

1. The model system needs to be separable into three components;
   a solute, an inner region around the solute, and an outer
   region separated from the solute by the inner region.
2. The solute can be a particle or molecule, in which case the
   SAFIRES boundary will expand spherically around the solute
   particle, or it can be a periodic surface model.
3. The solute needs to be fixed and cannot move during the simulation.
4. If the solute is a periodic surface model, the surface needs to
   be in the *xy* plane. Other configurations are currently not
   supported.
5. While the solute can be different, all particles in the inner and
   outer regions need to be chemically equivalent for SAFIRES to
   produce statistically correct results.

To learn about the mathematical justification and inner workings
of the SAFIRES method, refer to the
:doi:`the method paper for more info <10.1021/acs.jctc.1c00522>`.

Preparation of the Lennard-Jones model system
---------------------------------------------

In the following, we will create and equilibrate a Lennard-Jones
liquid model system from scratch for later use with SAFIRES.
We will be using argon parameters published
:doi:`here <10.1103/PhysRev.136.A405>`:

.. math:: T = 94.4 K,\
          \rho = 1.374\ g\ cm^{-3},\
          \sigma = 3.4\ Å,\
          \epsilon = 120\ k_\text{B}

Analogous to the approach explained in detail in the :ref:`qmmm`
tutorial, we first set up the atoms object based on a single atom
with a cell volume corresponding to the required density `\rho`::

    from ase import Atoms
    from ase import units

    atoms = Atoms('Ar', positions=[[0,0,0]])
    vol = (atoms.get_masses()[0] / units._Nav / (1.374 / 1e24))**(1 / 3.)
    atoms.cell = [vol, vol, vol]

Next, we'll want to repeat this unit cell until we reach the desired
size for the model system. To this end, the added particle is first
centered within its volume segment::

    atoms.center()
    atoms = atoms.repeat((8, 8, 8))
    atoms.pbc = [True, True, True]

We can now assign the Lennard Jones calculator::

    from ase.calculators.lj import LennardJones

    atoms.calc = LennardJones(epsilon=120 * units.kB,
                              sigma=3.4)

Lastly, we prepare the dynamics object for this simulation. To
equilibrate this liquid under the given conditions (fixed temperature),
an *NVT* ensemble calculation is required. The Vanden-Eijnden /
Ciccotti Langevin propagator (:class:`~ase.md.Langevin`) can be used to
this end. A time step of 1 fs is appropriate for this simulation.
The Langevin class comes with its on logging method, which we will use
to monitor the temperature convergence of this equilibration simulation.
We will also periodically save the trajectory for possible future use of
the equilibrated model system in other simulations::

    from ase.md import Langevin
    from ase.io.trajectory import Trajectory

    md = Langevin(atoms, timestep=1 * units.fs, temperature_K=94.4,
                  friction=0.01, logfile='preopt.log')
    traj_preopt = Trajectory('preopt.traj', 'w', atoms)
    md.attach(traj_preopt.write, interval=100)

    md.run(2000)

At the time of writing and using these settings, temperature
equilibration of this system is is achieved within 1-2 ps.
This simulation should take less than a minute on a modern CPU. It is
advisable, however, to run the equilibration for longer in order to
generate a set of uncorrelated, equilibrated starting configurations
that can be used for parallel production runs. This approach is a great
workaround to parallelize simulations that don't natively support
parallel computing, such as the LennardJones class in this particular
case.

Having obtained an equilibrated starting configuration, it is time to
prepare the actual SAFIRES run. Here, we assume that this follow up
run is part of the same input script and we can thus get to work on
the existing atoms object, which now contains the results of the
previous thermalization.

SAFIRES uses the tag system to differentiate between the solute and
the inner and outer region. To start preparations on the atoms object,
we first wrap the object and then assign ``atom.tag = 2`` to all
particles, which corresponds to the outer region. The solute and inner
region will be expanded from this in a subsequent step::

    atoms.wrap()
    atoms.set_tags(2)

For this example, we will set one Lennard Jones particle as the solute
and then expand the inner region around this atom, up to 5 % of the
total number of particles. Note that while SAFIRES is set up to handle
periodic boundary conditions, it is safest to make sure that the
flexible boundary is far away from the periodic boundary. Thus, we
calculate which particle is closest to the center of the simulation
box, set this as the solute (``atom.tag = 0``) and fix constrain it::

    import numpy as np
    from operator import itemgetter

    center = atoms.cell.diagonal() / 2
    distances = [[np.linalg.norm(atom.position - center), atom.index]
                 for atom in atoms]
    index_c = sorted(distances, key=itemgetter(0))[0][1]
    atoms[index_c].tag = 0

Note that ``np.linalg.norm()`` does not respect the periodic boundary
conditions but this is irrelevant in this case. Unlike in the next
part, where we expand the inner region around the central particle::

    ninner = int(len(atoms) * 0.05) + 1 # +1 for the solute
    distances = [[atoms.get_distance(index_c, atom.index, mic=True), atom.index]
                 for atom in atoms]
    distances = sorted(distances, key=itemgetter(0))
    for i in range(ninner + 1):
        # Start counting from i+1 to ignore the solute, which
        # is on top of this list with a distance of zero.
        atoms[distances[i+1][1]].tag = 1
    
We now need to rearrange the atoms object in a certain way. SAFIRES
requires that the solute (tag = 0) must always come first in the
atoms object. The inner and outer region particles / molecules can
be added afterwards in arbitrary order::
    newatoms = Atoms()
    newatoms.extend(atoms[[atom.index for atom in atoms
                           if atom.tag == 0]])
    newatoms.extend(atoms[[atom.index for atom in atoms
                           if atom.tag in [1,2]]])
    newatoms.cell = atoms.cell
    newatoms.pbc = atoms.pbc
    newatoms.calc = atoms.calc
    atoms = newatoms

Finally, the central particle is constrained. At the time of
writing this tutorial, SAFIRES requires that a particle or
molecule is designated as the origin (tag = 0) and that the
center of mass of the origin is frozen. It is possible in 
principle to define a ghost atom, which does not take part in
the chemistry of the simulation, as the origin instead. However,
for the sake of simplicity, we will simply constrain the central
LJ particle and use it as the origin. After the earlier
rearrangement, this particle has index 0::

    from ase.constraints import FixAtoms
        
    atoms.constraints = [FixAtoms(indices=[0])]

Now that SAFIRES will know which particle belongs to which region,
we can prepare the dynamics object for the SAFIRES calculation.
SAFIRES is fully energy conserving, and to demonstrate this fact
we will perform a *NVE* simulation using the Velocity Verlet
dynamics class (:class:`~ase/md/verlet/VelocityVerlet`)::

    from ase.md.verlet import VelocityVerlet

    md = VelocityVerlet(atoms, timestep=1 * units.fs)

After initializing the dynamics object, SAFIRES can be initialized
and appended to it. Here, ``natoms`` communicates to SAFIRES how
many atoms are in each solvent molecule (here: only 1)::

    from ase.md.safires import SAFIRES

    boundary = SAFIRES(atoms, mdobject=md, natoms=1)
    md.attach(boundary.safires, interval=1)

The interval must be set to 1 (every iteration), otherwise SAFIRES
will not properly fulfill its intended purpose.

.. note::
    SAFIRES will change the atomic configuration and re-calculate
    energy results in order to enforce the boundary.
    Thus, the logger and trajectory objects need to be
    appended to the dynamics object *after* SAFIRES in order for
    them to save the correct information.

Finally, we would like to save the trajectory and MD results into
files again. The :class:`VelocityVerlet` class supports trajectory writing
and logging. However, since SAFIRES will perform its work after a
successful dynamics iteration and will potentially undo and change
the trajectory and energy calculations in order to enforce the
flexible boundary, we cannot use the built in functionality. Instead,
we use the :class:`~ase/md/MDLogger` class to log the dynamics results
and append a new trajectory object::

    from ase.md import MDLogger

    traj_safires = Trajectory('safires.traj', 'w', atoms)
    md.attach(traj_safires.write, interval=1)

    logger = MDLogger(md, atoms, 'safires.log', mode='w')
    md.attach(logger, interval=1)

    md.run(1000)

A complete input script for this tutorial can be found under
/ase/doc/tutorials/safires/safires-lj-liquid.py.

A good way to judge the performance of the SAFIRES method is
to compare a run without SAFIRES (but fixed solute) with a
simulation using SAFIRES. When sampling the RDFs for these
model systems between the solute and all other particles,
the RDF distance `r` will correspond to the distance from
the solute. For this particlar example, SAFIRES will reproduce
exactly the RDF of the unconstrained simulation, see the
:doi:`the method paper <XX.XXXX/acs.jctc.XXXXXXX>`.
However, it is good practice to repeat this test for any new
system and combination of potentials to see the effect of the
boundary on the given system.

If you want to reproduce this RDF test, note that a lot of
uncorrelated configuration are necessary due to the specific
way the RDF is sampled. 1,000,000 iterations, sampled every
0.1 ps, will results in a smooth RDF for this particular 
example.
