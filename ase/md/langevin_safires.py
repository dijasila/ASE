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
        # Keeping track of collisions.
        self.recent = []


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

        # get_center_of_mass() breaks with certain constraints.
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
        # we can now reapply the constraints to the original
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

    
    def extrapolate_dt(self, atoms, forces, ftr_boundary_idx, 
                       outer_idx, checkup):
        """Return the time step required to resolve boundary event.

        Keyword arguments:
        previous_boundary_idx -- atom index of the inner region
                                 particle that defined the boundary
                                 during the previous iteration
        boundary_idx -- atom index of the inner region particle that
                        defines the boundary on the current iteration
        outer_idx -- atom index of the outer region particle that
                     triggered the boundary event
        checkup -- True/False, is used internally to indicate if
                   this is a checkup run which occurs after a
                   successful boundary event resolution. this is done
                   to catch rare cases where a secon outer region
                   particle has entered the inner region during the
                   resolution of a first boundary event. in this case,
                   slightly different rules apply.
        """

        # results dict
        res = {}  # structure: {inner particle index: extrapolated factor}
            
        # Extrapolation uses the center of mass atoms object instead
        # of the "real" one.
        com_atoms, com_forces, r, d, boundary_idx, boundary = (
                self.update(atoms, forces))

        # convert list to set (no duplicate values)
        for inner_idx in set([boundary_idx, ftr_boundary_idx]):
            # find point where outer and inner region particles
            # have the same distance from COM, i.e. both are on
            # the boundary. we need to propagate both particles
            # to this point to perform an exact elastic collision.

            if inner_idx in self.recent and outer_idx in self.recent:
                # First, check if inner and outer performed a collision
                # in the very last iteration. This can occur if
                # a collision is performed and in the subsequent
                # iteration (within remaining_dt) the same outer
                # collides with another inner. If we do the
                # extrapolation based on both these inner, the inner
                # from the previous iteration will always give
                # the smaller dt because this inner and the outer are
                # both on the boundary at the state of checking.
                # We need to ignore this pair since it's already
                # been resolved.
                continue

            # retreive forces, velocities, distances, masses
            # for image before the event
            T = self.temp
            fr = self.fr
            r_outer = r[outer_idx]
            r_inner = r[inner_idx]
            m_outer = com_atoms[outer_idx].mass
            v_outer = com_atoms[outer_idx].momentum / m_outer
            f_outer = com_forces[outer_idx] / m_outer
            m_inner = com_atoms[inner_idx].mass
            v_inner = com_atoms[inner_idx].momentum / m_inner
            f_inner = com_forces[inner_idx] / m_inner

            # if molecules are used, which are reduced to
            # pseudoparticles with properties centered on their COM,
            # we need to re-expand the pseudoparticle indices into the
            # "real" indices by multiplying the pseudoparticle index by
            # the number of atoms in each molecule.
            # furthermore, shift in the indexing due to the solute or
            # periodic surface model (which can have arbitrary number
            # of atoms) needs to be accounted for.
            outer_real = self.idx_real[outer_idx]
            inner_real = self.idx_real[inner_idx]

            print("r_outer = ", r_outer)
            print("r_inner = ", r_inner)
            print("m_outer = ", m_outer)
            print("m_inner = ", m_inner)
            print("v_inner = ", v_inner)
            print("v_outer = ", v_outer)
            print("f_outer = ", f_outer)
            print("f_inner = ", f_inner)
            print("outer_idx = ", outer_idx)
            print("inner_id = ", inner_idx)
            print("outer_real = ", outer_real)
            print("inner_real = ", inner_real)

            # if inner/outer particles are molecules
            if self.nout > 1 or self.nin > 1:
                m_outer_list = [math.sqrt(xm) for xm in
                                self.masses[outer_real:outer_real + self.nout]]
                m_inner_list = [math.sqrt(xm) for xm in
                                self.masses[inner_real:inner_real + self.nin]]
                xi_outer = (np.dot(m_outer_list,
                            self.xi[outer_real:outer_real + self.nout])
                            / m_outer)
                xi_inner = (np.dot(m_inner_list,
                            self.xi[inner_real:inner_real + self.nin])
                            / m_inner)
                eta_outer = (np.dot(m_outer_list,
                             self.eta[outer_real:outer_real + self.nout])
                             / m_outer)
                eta_inner = (np.dot(m_inner_list,
                             self.eta[inner_real:inner_real + self.nin])
                             / m_inner)

                # Need to expand this since it can be an array of fr values
                # CHECK HERE IF FR is ARRAY
                sig_outer = math.sqrt(2 * T * fr)
                sig_inner = math.sqrt(2 * T * fr)

            # if inner/outer particles are monoatomic
            else:
                xi_outer = self.xi[outer_real]
                xi_inner = self.xi[inner_real]
                eta_outer = self.eta[outer_real]
                eta_inner = self.eta[inner_real]
                sig_outer = math.sqrt(2 * T * fr / m_outer)
                sig_inner = math.sqrt(2 * T * fr / m_inner)
                print("xi_outer = ", xi_outer)
                print("xi_inner = ", xi_inner)
                print("eta_outer = ", eta_outer)
                print("eta_inner = ", eta_inner)
                print("sig_outer = ", sig_outer)
                print("sig_inner = ", sig_inner)

            # surface calculations: we only need z components
            if self.surface:
                v_outer[0] = 0.
                v_outer[1] = 0.
                v_inner[0] = 0.
                v_inner[1] = 0.
                f_outer[0] = 0.
                f_outer[1] = 0.
                f_inner[0] = 0.
                f_inner[1] = 0.
                xi_outer[0] = 0.
                xi_outer[1] = 0.
                xi_inner[0] = 0.
                xi_inner[1] = 0.
                eta_outer[0] = 0.
                eta_outer[1] = 0.
                eta_inner[0] = 0.
                eta_inner[1] = 0.

            # the time step extrapolation is based on solving a
            # 2nd degree polynomial of the form:
            # y = c0*x^2 + c1*x + c2.
            # a and b are velocity modifiers derived from the first
            # velocity half step in the Langevin algorithm. see
            # publication for more details.
            if not checkup:
                idt = self.dt
                sqrt_3 = math.sqrt(3)
                a_outer = (idt * (f_outer - fr * v_outer) / 2
                           + math.sqrt(idt) * sig_outer * xi_outer / 2
                           - idt * idt * fr * (f_outer - fr * v_outer) / 8
                           - idt**1.5 * fr * sig_outer * (xi_outer / 2
                           + eta_outer / sqrt_3 / 4))
                b_outer = math.sqrt(idt) * sig_outer * eta_outer / (2 * sqrt_3)

                a_inner = (idt * (f_inner - fr * v_inner) / 2
                           + math.sqrt(idt) * sig_inner * xi_inner / 2
                           - idt * idt * fr * (f_inner - fr * v_inner) / 8
                           - idt**1.5 * fr * sig_inner * (xi_inner / 2
                           + eta_inner / sqrt_3 / 4))
                b_inner = math.sqrt(idt) * sig_inner * eta_inner / (2 * sqrt_3)

                print("a_outer = ", a_outer)
                print("a_inner = ", a_inner)
                print("b_outer = ", b_outer)
                print("b_inner = ", b_inner)

            else:
                a_outer = 0
                a_inner = 0
                b_outer = 0
                b_inner = 0

            v_outer += a_outer
            v_outer += b_outer
            v_inner += a_inner
            v_inner += b_inner

            # set up polynomial coefficients
            c0 = np.dot(r_inner, r_inner) - np.dot(r_outer, r_outer)
            c1 = 2 * np.dot(r_inner, v_inner) - 2 * np.dot(r_outer, v_outer)
            c2 = np.dot(v_inner, v_inner) - np.dot(v_outer, v_outer)

            print("c0 = ", c0)
            print("c1 = ", c1)
            print("c2 = ", c2)

            # find roots
            roots = np.roots([c2, c1, c0])
            #self.debuglog("   < TIME STEP EXTRAPOLATION >\n"
            #              "   all extrapolated roots: {:s}\n"
            #              .format(np.array2string(roots)))

            for val in roots:
                if np.isreal(val) and val <= self.dt and val > 0:
                    # the quadratic polynomial yields four roots.
                    # we're only interested in the SMALLEST positive real
                    # value, which is the required time step.
                    res.update({inner_idx: np.real(val)})

                    #if self.debug:
                    #    # debug logging
                    #    r_outer_new = r_outer + val * v_outer
                    #    d_outer_new = np.linalg.norm(r_outer_new)
                    #    r_inner_new = r_inner + val * v_inner
                    #    d_inner_new = np.linalg.norm(r_inner_new)
                    #    self.debuglog("   d_inner extrapolated = {:.12f}\n"
                    #                  .format(d_inner_new))
                    #    self.debuglog("   d_outer extrapolated = {:.12f}\n"
                    #                  .format(d_outer_new))
                    #    self.debuglog("   Extapolated dt for atom pair {:d}"
                    #                  " (INNER) - {:d} (OUTER): {:.5f}\n"
                    #                  .format(inner_idx, outer_idx,
                    #                          np.real(val)))

        if not res:
            # if none of the obtained roots fit the criteria (real,
            # positive, <= initial time step), then we have a problem.
            # this is indicative of a major issue.
            error = ("ERROR:\n\n"
                     "Unable to extrapolate time step (all real roots\n"
                     "<= 0 or > default time step).\n"
                     "Roots: {:s}\n".format(np.array2string(roots)))
            #self.debuglog(error)
            #self.debugtraj()
            raise SystemExit(error)
        else:
            return res


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
        v = atoms.get_velocities()
        f = forces / m
        T = self.temp
        fr = self.fr
        xi = self.xi
        eta = self.eta
        sig = np.sqrt(2 * T * fr / m)
        sqrt_3 = math.sqrt(3)

        # pre-calculate (random) force constant
        # based on default time step
        idt = self.dt
        if not checkup:
            c = (idt * (f - fr * v) / 2
                 + math.sqrt(idt) * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_3) / 4)
            d = math.sqrt(idt) * sig * eta / (2 * sqrt_3)
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
            # Velocity update occurs at the end of step(),
            # after force update.
            if constraints == True:
                atoms.set_positions(x + dt * v)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)

        return atoms

    def step(self, forces=None):
        atoms = self.atoms
        lenatoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(lenatoms, 3))
        self.eta = self.rng.standard_normal(size=(lenatoms, 3))

        # To keep the center of mass stationary, the random arrays should add to (0,0,0)
        if self.fix_com:
            self.xi -= self.xi.sum(axis=0) / lenatoms
            self.eta -= self.eta.sum(axis=0) / lenatoms
        
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
        future_atoms = self.propagate(atoms.copy(), forces, self.dt, checkup=False,
                halfstep=1, constraints=True)
        
        # Convert atoms and future_atoms to COM atoms objects.
        ftr_com_atoms, ftr_forces, ftr_r, ftr_d, ftr_boundary_idx, ftr_boundary = (
                self.update(future_atoms, forces))

        # Check if future_atoms contains boundary conflict.
        conflicts = [(ftr_boundary_idx, atom.index) for atom in ftr_com_atoms
                     if atom.tag == 3 and ftr_d[atom.index] < ftr_boundary]

        # If there are boundary conflicts, execute problem solving.
        if conflicts:
            print("CONFLICTS = ", conflicts)
            print("regular dt = ", self.dt)
            
            # Find the conflict that occurs earliest in time
            dt_list = [self.extrapolate_dt(atoms, forces, c[0], c[1], 
                       checkup=False) for c in conflicts]

            print("dt_list = ", dt_list)
            raise SystemExit()
    
        # No conflict: we can use future_atoms right away.
        else:
            atoms = self.propagate(atoms, forces, self.dt, checkup=False,
                halfstep=1, constraints=True)

            # Finish propagation as usual.
            m = self.masses
            v = self.v
            T = self.temp
            fr = self.fr
            xi = self.xi
            eta = self.eta
            sig = np.sqrt(2 * T * fr / m)
            sqrt_of_3 = math.sqrt(3)

            forces = atoms.get_forces(md=True)
            f = forces / m
            c = (self.dt * (f - fr * v) / 2
                 + math.sqrt(self.dt) * sig * xi / 2
                 - self.dt * self.dt * fr * (f - fr * v) / 8
                 - self.dt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            v += c
            atoms.set_momenta(v * m)

        return forces
