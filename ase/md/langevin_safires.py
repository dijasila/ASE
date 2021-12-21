"""Langevin dynamics class."""

import numpy as np
from operator import itemgetter
import math

from ase import Atoms
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
from ase.geometry import find_mic


class Langevin(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 4

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 fixcm=True, *, temperature_K=None, trajectory=None,
                 logfile=None, loginterval=1, communicator=world,
                 rng=None, append_trajectory=False, surface=False,
                 reflective=False, natoms=None, natoms_in=None):
        """
        Parameters:

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float (deprecated)
            The desired temperature, in electron volt.

        temperature_K: float
            The desired temperature, in Kelvin.

        friction: float
            A friction coefficient, typically 1e-4 to 1e-2.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object (optional)
            Random number generator, by default numpy.random.  Must have a
            standard_normal method matching the signature of
            numpy.random.standard_normal.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* (the default) for no
            trajectory.

        communicator: MPI communicator (optional)
            Communicator used to distribute random numbers to all tasks.
            Default: ase.parallel.world. Set to None to disable communication.

        append_trajectory: bool (optional)
            Defaults to False, which causes the trajectory file to be
            overwritten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        The temperature and friction are normally scalars, but in principle one
        quantity per atom could be specified by giving an array.

        RATTLE constraints can be used with these propagators, see:
        E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

        The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
        of the above reference.  That reference also contains another
        propagator in Eq. 21/34; but that propagator is not quasi-symplectic
        and gives a systematic offset in the temperature at large time steps.
        """
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')
        self.fix_com = fixcm
        if communicator is None:
            communicator = DummyMPI()
        self.communicator = communicator
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        
        # SAFIRES specific stuff
        self.nout = natoms
        if natoms_in is not None:
            self.nin = natoms_in
        else:
            self.nin = natoms
        self.reflective = reflective
        self.surface = surface
        self.constraints = atoms.constraints.copy()
        # Relative index of pseudoparticle in total atoms object
        self.idx_real = None
        self.nsol = len([atom.index for atom in atoms
                       if atom.tag == 1])
        # Set up natoms array according to tag : 0, 1, 2, 3
        self.nall = np.array([1, self.nsol, self.nin, self.nout])


        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)
    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'friction': self.fr,
                  'fixcm': self.fix_com})
        return d

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')

    def set_friction(self, friction):
        self.fr = friction

    def set_timestep(self, timestep):
        self.dt = timestep

    def update(self, atoms, forces):
        """Return reduced pseudoparticle atoms object.

        Keyword arguments:
        atoms -- ASE atoms object with attached calculator containing
                 atomic positions, cell information, and results from
                 an MD iteration

        Return values:
        com_atoms -- if used with molecules: com_atoms contains new
                     pseudoparticles centered on the COM of the
                     original molecule with effective
                     velocities for the whole molecule. if monoatomic
                     species are used, this will be identical to the
                     original atoms object.
        forces -- effective force on each pseudoparticle. forces need
                  to be stored separate from the atoms object. if
                  monoatomic species are used, this array is identical
                  to that returned from the calculator.
        r -- array of vectors between the COM of the solute and all
             inner and outer region (pseudo-) particles in the system
        d -- array of absolute distances between the COM of the solute
             and all inner and outer region (pseudo-) particles
             in the system
        boundary_idx -- atom object index of the inner region (pseudo-)
                        particle that is furthest away from the COM of
                        the solute, defining the flexible boundary
        boundary -- absolute distance between the inner region (pseudo-)
                    particle and the COM of the solute, defining the
                    radius of the boundary sphere for molecular solutes
                    or the distance of the boundary plane from the
                    solute for surfaces.

        Arrays r and d have the same ordering as atoms object.
        """

        # calculate distance of the resulting COM
        i = 0
        com_atoms = Atoms()
        com_atoms.pbc = atoms.pbc
        com_atoms.cell = atoms.cell

        # need to make sure constraints are off
        # get_com method breaks GPAW fast FixBondLength
        # constraints (-> for rigid H2O)
        self.constraints = atoms.constraints.copy()
        atoms.constraints = []

        # calculate cumulative properties for all inner/outer
        # region particles. for monoatomic inner/outer particles,
        # a 1:1 copy is created.
        com_forces = []
        # This should be done elsewhere since it does not change
        # during the simulation
        idx_real = []

        while i < len(atoms):
            idx = atoms[i].index
            tag = atoms[i].tag
            nat = self.nall[tag]
            com = atoms[idx:idx + nat].get_center_of_mass()
            M = np.sum(self.masses[idx:idx + nat])
            mom = np.sum(atoms[idx:idx + nat].get_momenta(), axis=0)
            frc = np.sum(forces[idx:idx + nat], axis=0)
            sym = atoms[idx].symbol

            if tag == 1:
                sol_com = com

            # Create new atoms
            tmp = Atoms(sym)
            tmp.set_positions([com])
            tmp.set_momenta([mom])
            tmp.set_masses([M])
            tmp.set_tags([tag])
            com_forces.append(frc)

            # Append and iterate
            com_atoms += tmp
            idx_real.append(i)
            i += nat

        if self.surface:
            # we only need z coordinates for surface calculations
            for atom in com_atoms:
                atom.position[0] = 0.
                atom.position[1] = 0.

        self.idx_real = idx_real
        # we can no reapply the constraints to the original
        # atoms object. all further processing will be done
        # on the pseudoparticle com_atoms object, which does
        # not have constraints
        atoms.constraints = self.constraints.copy()

        # calculate absolute distances and distance vectors between
        # COM of solute and all inner and outer region particles
        # (respect PBCs in distance calculations)
        r, d = find_mic([atom.position for atom in com_atoms] - sol_com,
                       com_atoms.cell, com_atoms.pbc)

        # list all particles in the inner region
        inner_mols = [(atom.index, d[atom.index])
                      for atom in com_atoms if atom.tag == 2]

        # boundary is defined by the inner region particle that has
        # the largest absolute distance from the COM of the solute
        boundary_idx, boundary = sorted(inner_mols, key=itemgetter(1),
                                        reverse=True)[0]

        return com_atoms, com_forces, r, d, boundary_idx, boundary


    def propagate(self, atoms, forces, dt, checkup, halfstep, 
                  constraints=True):
        """Propagate the simulation.

        Keyword arguments:
        dt -- the time step used to propagate.
        checkup -- True/False, is used internally to indicate if
                   this is a checkup run which occurs after a
                   successful boundary event resolution. this is done
                   to catch rare cases where a second outer region
                   particle has entered the inner region during the
                   resolution of a first boundary event.
        halfstep -- 1/2, indicates if we're propagating so that the
                    conflicting inner and outer particle are at the
                    same distance from the center using the
                    extrapolated time step from extrapolate_dt()
                    to perform a collision (halfstep = 1) or if the
                    collision is already performed and we're
                    propagating to make up the remaining time to
                    complete a full default time step (halfstep = 2).
        """

        # retreive parameters
        x = atoms.get_positions()
        m = self.masses
        v = self.v
        f = forces / m
        T = self.temp
        fr = self.fr
        xi = self.xi
        eta = self.eta
        sig = np.sqrt(2 * T * fr / m)
        sqrt_of_3 = math.sqrt(3)

        # pre-calculate (random) force constant
        # based on default time step
        idt = self.dt
        if not checkup:
            c = (idt * (f - fr * v) / 2
                 + math.sqrt(idt) * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            d = math.sqrt(idt) * sig * eta / (2 * sqrt_of_3)
        else:
            # if checkup is True, this means we already performed an entire
            # propagation cycle and have already updated the velocities
            # based on values for the full default time step. thus we
            # need to make sure not to update velocities a second time
            # because that would destroy energy conservation.
            c = np.asarray([np.asarray([0., 0., 0.]) for atom in self.atoms])
            d = np.asarray([np.asarray([0., 0., 0.]) for atom in self.atoms])

        if halfstep == 1:
            # friction and (random) forces should only be
            # applied during the first halfstep.
            v += c + d
            if constraints == True:
                atoms.set_positions(x + dt * v)
                v = (atoms.get_positions() - x) / dt
                atoms.set_momenta(v * m)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)
                v = (atoms.get_positions() - x) / dt
                atoms.set_momenta(v * m, apply_constraint=False)

        if halfstep == 2:
            # at the end of the second part of the time step,
            # do the force update and the second
            # velocity halfstep
            if constraints == True:
                atoms.set_positions(x + dt * v)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)
            #v = (atoms.get_positions() - x - dt * d) / dt
            #f = atoms.get_forces(md=True) / m
            ##c = (idt * (f - fr * v) / 2
            #     + math.sqrt(idt) * sig * xi / 2
            #     - idt * idt * fr * (f - fr * v) / 8
            #     - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            #v += c
            #atoms.set_momenta(v * m)

        return atoms

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(natoms, 3))
        self.eta = self.rng.standard_normal(size=(natoms, 3))

        # To keep the center of mass stationary, the random arrays should add to (0,0,0)
        if self.fix_com:
            self.xi -= self.xi.sum(axis=0) / natoms
            self.eta -= self.eta.sum(axis=0) / natoms

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, self.xi, rand=True)
                constraint.redistribute_forces_md(atoms, self.eta, rand=True)

        self.communicator.broadcast(self.xi, 0)
        self.communicator.broadcast(self.eta, 0)

        """
            PLAN FOR LANGEVIN_SAFIRES IMPLEMENTATION
            
            1) future_atoms = safires_propagate(atoms, NO CONSTRAINTS, halfstep=1, dt=dt)
            2) check if future_atoms causes issues
            3a) issues detected:
                i) extrapolate smallest frac_dt
                ii) safires_propagate(atoms, NO CONSTRAINTS, halfstep=1, dt=frac_dt)
                iii) safires_collide(atoms, inner_conflict, outer_conflict)
                iv) safires_propagate(atoms, constraints, halfstep=2, dt=dt-frac_dt)
                v) go back to 2 with future_atoms = atoms
            3b) no issues detected: update forces

        """
        # Propagate a copy of the atoms object by self.dt.
        future_atoms = self.propagate(atoms, forces, self.dt, checkup=False,
                halfstep=1, constraints=False)
        
        # Convert atoms and future_atoms to COM atoms objects.
        com_atoms, forces, r, d, boundary_idx, boundary = (
                self.update(future_atoms, forces))

        # Check if future_atoms contains boundary conflict.
        conflicts = []
        for atom in [atom for atom in com_atoms if atom.tag == 3]:
            if d[atom.index] < boundary:
                conflicts.append((boundary_idx, atom.index))

        # Check if conflicts were detected, otherwise propagate as usual.
        if conflicts:
            raise SystemExit("detected problem, exiting")
        else:
            m = self.masses
            v = self.v
            T = self.temp
            fr = self.fr
            xi = self.xi
            eta = self.eta
            sig = np.sqrt(2 * T * fr / m)
            sqrt_of_3 = math.sqrt(3)

            atoms = future_atoms
            forces = atoms.get_forces(md=True) / m
            c = (self.dt * (forces - fr * v) / 2
                 + math.sqrt(self.dt) * sig * xi / 2
                 - self.dt * self.dt * fr * (forces - fr * v) / 8
                 - self.dt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            v += c
            atoms.set_momenta(v * m)

        return forces
