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
   Fixing only the solute center of mass is possible as well.
4. If the solute is a periodic surface model, the surface needs to
   be in the *xy* plane. Other configurations are currently not
   supported.
5. While the solute can be different, all particles in the inner and
   outer regions need to be chemically equivalent for SAFIRES to
   produce statistically correct results.

To learn about the mathematical justification and inner workings
of the SAFIRES method, refer to
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

Temperature equilibration of this system should be achieved within 1-2 
ps. This simulation should take less than a minute on a modern CPU. It
is advisable, however, to run the equilibration for longer in order to
generate a set of uncorrelated, equilibrated starting configurations
that can be used for parallel production runs.

Having obtained an equilibrated starting configuration, it is time to
prepare the actual SAFIRES run. Here, we assume that this follow-up
run is part of the same input script and we can thus get to work on
the existing atoms object, which now contains the results of the
previous thermalization.

SAFIRES uses the tag system to differentiate between the solute and
the inner and outer region. To start preparations on the atoms object,
we first wrap the object and then assign ``atom.tag = 3`` to all
particles, which corresponds to the outer region. The solute and inner
region will be expanded from this in a subsequent step::

    atoms.wrap()
    atoms.set_tags(3)

For this example, we will set one Lennard Jones particle as the solute
and then expand the inner region around this atom, up to 5 % of the
total number of particles. Note that while SAFIRES is set up to handle
periodic boundary conditions, it is safest to make sure that the
flexible boundary is far away from the periodic boundary. Thus, we
calculate which particle is closest to the center of the simulation
box, set this as the solute (``atom.tag = 1``) and fix constrain it::

    import numpy as np
    from operator import itemgetter

    center = atoms.cell.diagonal() / 2
    distances = [[np.linalg.norm(atom.position - center), atom.index]
                 for atom in atoms]
    index_c = sorted(distances, key=itemgetter(0))[0][1]
    atoms[index_c].tag = 1

Note that ``np.linalg.norm()`` does not respect the periodic boundary
conditions but this is irrelevant in this case. Unlike in the next
part, where we expand the inner region around the central particle::

    ninner = int(len(atoms) * 0.05) + 1 # + 1 for the solute
    distances = [[atoms.get_distance(index_c, atom.index, mic=True), atom.index]
                 for atom in atoms]
    distances = sorted(distances, key=itemgetter(0))
    for i in range(ninner + 1):
        # Start counting from i + 1 to ignore the solute, which
        # is on top of this list with a distance of zero.
        atoms[distances[i+1][1]].tag = 2
    
Finally, the central particle, which has the index ``index_c``, will
be constrained::

    from ase.constraints import FixAtoms
        
    atoms.constraints = [FixAtoms(indices=[index_c])]

Now that SAFIRES will know which particle belongs to which region,
we can prepare the dynamics object for the SAFIRES calculation.
SAFIRES is fully energy conserving, and to demonstrate this fact
we will perform a *NVE* simulation. The SAFIRES class is derived
from :class:`~ase.md.Langevin` and accepts the same parameters.
To execute a *NVE* run, the ``friction`` parameter must be set to zero.
The ``temperature_K`` parameter must be set as well as per requirement
but will not affect the simulation with zero ``friction``::

    from ase.md.safires import SAFIRES

    md = SAFIRES(atoms, timestep=1 * units.fs, friction=0,
             temperature_K=0, natoms=1, logfile="md.log")

``natoms`` communicates to SAFIRES how many atoms are in each solvent
molecule (here: only 1) and ``logfile`` names the output file of the
built-in :class:`~ase/md/MDLogger` class that SAFIRES uses.

Finally, we append a trajectory object to the dynamics and start
running the simulation::

    traj_md = Trajectory('md.traj', 'w', atoms)
    md.attach(traj_md.write, interval=1)

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
:doi:`the method paper <10.1103/PhysRev.136.A405>`.
However, it is good practice to repeat this test for any new
system and combination of potentials to see the effect of the
boundary on the given system.

If you want to reproduce this RDF test, note that a lot of
uncorrelated configuration are necessary due to the specific
way the RDF is sampled. 1,000,000 iterations, sampled every
0.1 ps, will results in a smooth RDF for this particular 
example.
