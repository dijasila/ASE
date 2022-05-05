"""SAFIRES dynamics class.

This Langevin-derived dynamics class is intended for hybrid
calculations where an inner and an outer region need to be
kept separated. SAFIRES avoids boundary crossing of molecules
by performing elastic collisions between their centers of mass
at the boundary in an energy-conserving fashion.

When using SAFIRES, cite:
J. Chem. Theory Comput. 2021, 17, 9, 5863–5875
"""

import numpy as np
import math
import warnings
from operator import itemgetter

from ase import Atoms
from ase.md.md import MolecularDynamics
from ase.parallel import world, DummyMPI
from ase import units
from ase.geometry import find_mic

_allowed_constraints = {'FixAtoms', 'FixCom'}

class SAFIRES(MolecularDynamics):
    """SAFIRES (constant N, V, T, variable dt) molecular dynamics."""

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 natoms=None, natoms_in=None, fixcm=True, *, 
                 temperature_K=None, trajectory=None, logfile=None, 
                 loginterval=1, communicator=world, rng=None, 
                 append_trajectory=False, surface=False, reflective=False):
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

        natoms: int
            SAFIRES parameter that determines the number of atoms of
            each solvent molecule (tag in [2, 3]).

        natoms_in: int (optional)
            SAFIRES parameter that determines the number of atoms of
            each solvent molecule in the inner region (if different
            from the outer region; tag == 2).

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

        surface: bool (optional)
            Defaults to False, changes SAFIRES behavior to expect either a
            molecular model system (False) or a periodic surface model
            system (True).

        reflective: bool (optional)
            Defaults to False, changes SAFIRES behavior to resolve boundary
            conflicts by elastic collisions involving momentum exchange
            between collision partners (False) or have the boundary act as
            a hard reflective surface (True).

        The temperature and friction are normally scalars, but in principle
        one quantity per atom could be specified by giving an array.

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
        self.remaining_dt = 0

        # Assertions / Warnings
        assert any(4 > atom.tag for atom in atoms), \
                   'Atom tag 4 and above not supported'
        assert any(1 == atom.tag for atom in atoms), \
                   'Solute is not tagged (tag=1) correctly'
        assert any(2 == atom.tag for atom in atoms), \
                   'Inner region is not tagged (tag=2) correctly'
        assert any(3 == atom.tag for atom in atoms), \
                   'Outer region is not tagged (tag=3) correctly'

        m_out = np.array([atom.mass for atom in atoms if atom.tag == 3])
        nm_out = int(len(m_out) / self.nout)

        m_in = np.array([atom.mass for atom in atoms if atom.tag == 2])
        nm_in = int(len(m_in) / self.nin)
        # Raise warning if masses of inner and outer solvent are different
        if not m_out.sum() / nm_out == m_in.sum() / nm_in:
            warnings.warn('The mass of inner and outer solvent molecules is \
                           not exactly the same')

        assert atoms.constraints, \
               'Constraints are not set (solute can not move)'

        assert self.check_constraints(atoms), \
               'Solute constraint not correctly set'

        # Final sanity check, make sure all inner are closer to 
        # origin than outer.
        assert self.check_distances(atoms), \
               'Outer molecule closer to origin than inner'

        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update({'temperature_K': self.temp / units.kB,
                  'friction': self.fr,
                  'fixcm': self.fix_com})
        return d

    def check_constraints(self, atoms):
        """ Check that solute has either FixAtoms or FixCom """
        sol_idx = np.array([atom.index for atom in atoms if atom.tag == 1])
        correct = np.array([False])
 
        for i, c in enumerate(atoms.constraints):
            if c.todict()['name'] in _allowed_constraints:
                correct = c.index == sol_idx
 
            if correct.all():
                break
 
        return correct.all()

    def check_distances(self, atoms):
        """
            Check that all outer atoms (tag == 3) are further away than
            all inner atoms (tag == 2).
        """

        cm_origin = atoms[[atom.index for atom 
                           in atoms if atom.tag == 1]].get_center_of_mass()
        # collect in/out
        cm_inner = []
        cm_outer = []

        i = 0
        while i < len(atoms):
            tag = atoms[i].tag
            idx = atoms[i].index
            nat = self.nall[tag]
            if tag == 2:
                cm_inner.append(atoms[idx:idx + nat].get_center_of_mass())
            if tag == 3:
                cm_outer.append(atoms[idx:idx + nat].get_center_of_mass())
            i += nat

        d_inner = np.linalg.norm(cm_inner - cm_origin, axis=1)
        d_outer = np.linalg.norm(cm_outer - cm_origin, axis=1)

        return max(d_inner) < min(d_outer)


    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(temperature,
                                                         temperature_K, 'eV')

    def set_friction(self, friction):
        self.fr = friction

    def set_timestep(self, timestep):
        self.dt = timestep

    def debuglog(self, content):
        print(content)

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def calc_angle(self, n1, n2):
        """Vincenty, T. Survey Review 23, 88–93 (1975)."""
        return(np.arctan2(np.linalg.norm(np.cross(n1, n2)), np.dot(n1, n2)))

    def rotation_matrix(self, axis, theta):
        """ Return rotation matrix for rotation by theta around axis.

        Return the rotation matrix associated with counterclockwise
        rotation about the given axis by theta radians.
        Euler-Rodrigues formula, code stolen from
        stackoverflow.com/questions/6802577.
        """
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def update(self, atoms, forces):
        """Return reduced pseudoparticle atoms object.

        Parameters:
        
        atoms: object
            ASE atoms object holding current atomic configuration.
        
        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        Return values:
        
        com_atoms: object
            If used with molecules: com_atoms contains new 
            pseudoparticles centered on the COM of the original
            molecule with effective velocities for the whole molecule.
            If monoatomic species are used, this will be identical to 
            the original atoms object.

        forces: array
            Numpy array containing the vectorfield of forces on
            all reduced COM pseudoparticles held by com_atoms.
        
        r: array
            Numpy array containing the field of distance vectors 
            between the COM of the solute and all inner and outer 
            region (pseudo-) particles in the system.
        
        d: array
            Numpy array containing the absolute distances between
            the COM of the solute and all inner and outer region
            (pseudo-) particles in the system.

        boundary_idx: int
            ASE Atoms object index of the inner region (pseudo-)
            particle that is furthest away from the COM of the solute,
            defining the flexible boundary.

        boundary: float
            Absolute distance between boundary_idx and the COM of the 
            solute, defining the radius of the boundary sphere for
            molecular solutes or the distance of the boundary plane
            from the solute for surfaces.

        Arrays forces, r, and d have the same ordering as atoms object.
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

        Parameters:

        atoms: object
            ASE atoms object holding the current configuration.

        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        ftr_boundary_idx: int 
            Atoms object index of the inner region particle that 
            defines the boundary during the conflict iteration.

        outer_idx int
            Atoms object index of the outer region particle that
            triggered the boundary event.
        
        checkup: bool
            False for first conflict resolution within a given time
            step, True for any subsequent conflict resolitions within
            the same time step.
        """

        # results dict
        res = []  # structure: (inner particle index, extrapolated factor)
            
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

            #print("r_outer = ", r_outer)
            #print("r_inner = ", r_inner)
            #print("m_outer = ", m_outer)
            #print("m_inner = ", m_inner)
            #print("v_inner = ", v_inner)
            #print("v_outer = ", v_outer)
            #print("f_outer = ", f_outer)
            #print("f_inner = ", f_inner)
            #print("outer_idx = ", outer_idx)
            #print("inner_id = ", inner_idx)
            #print("outer_real = ", outer_real)
            #print("inner_real = ", inner_real)

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
                
                #print("xi_outer = ", xi_outer)
                #print("xi_inner = ", xi_inner)
                #print("eta_outer = ", eta_outer)
                #print("eta_inner = ", eta_inner)
                #print("sig_outer = ", sig_outer)
                #print("sig_inner = ", sig_inner)

            # if inner/outer particles are monoatomic
            else:
                xi_outer = self.xi[outer_real]
                xi_inner = self.xi[inner_real]
                eta_outer = self.eta[outer_real]
                eta_inner = self.eta[inner_real]
                sig_outer = math.sqrt(2 * T * fr / m_outer)
                sig_inner = math.sqrt(2 * T * fr / m_inner)

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
                sqrt_idt = math.sqrt(idt)
                pwr_15_idt = idt**1.5

                a_outer = (idt * (f_outer - fr * v_outer) / 2
                           + sqrt_idt * sig_outer * xi_outer / 2
                           - idt * idt * fr * (f_outer - fr * v_outer) / 8
                           - pwr_15_idt * fr * sig_outer * (xi_outer / 2
                           + eta_outer / sqrt_3) / 4)
                b_outer = sqrt_idt * sig_outer * eta_outer / (2 * sqrt_3)

                a_inner = (idt * (f_inner - fr * v_inner) / 2
                           + sqrt_idt * sig_inner * xi_inner / 2
                           - idt * idt * fr * (f_inner - fr * v_inner) / 8
                           - pwr_15_idt * fr * sig_inner * (xi_inner / 2
                           + eta_inner / sqrt_3) / 4)
                b_inner = sqrt_idt * sig_inner * eta_inner / (2 * sqrt_3)
            
                #print("a_outer = ", a_outer)
                #print("a_inner = ", a_inner)
                #print("b_outer = ", b_outer)
                #print("b_inner = ", b_inner)

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

            #print("c0 = ", c0)
            #print("c1 = ", c1)
            #print("c2 = ", c2)

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
                    res.append((inner_idx, outer_idx, np.real(val)))

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
            error = ("\nERROR:\n"
                     "Unable to extrapolate time step.\n"
                     "All real roots <= 0 or > default time step).\n"
                     "Roots: {:s}\n".format(np.array2string(roots)))
            #self.debuglog(error)
            #self.debugtraj()
            raise SystemExit(error)
        else:
            # return smallest timestep
            return sorted(res, key=itemgetter(2))[0]


    def propagate(self, atoms, forces, dt, checkup, halfstep, 
                  constraints=True):
        """Propagate the simulation.

        Parameters:

        atoms: object
            ASE atoms object holding the current configuration.

        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        dt: float
            The time step used to propagate.

        checkup: bool
            False for first conflict resolution within a given time
            step, True for any subsequent conflict resolitions within
            the same time step.

        halfstep: int (1 or 2)
            Boundary conflict resolution is performed in two halfsteps.
            1: propagate by fraction of dt so that outermost inner 
               particle and innermost outer particle have the same 
               distance from the solute, redirect particles.
            2: after conflict resolution, propagate remaining
               fraction of dt to complete a full default time step.

        constraints: bool
            Defaults to True, turn constraints on or off during the
            propagation.
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
        idt = self.dt
        sqrt_idt = math.sqrt(idt)

        # pre-calculate (random) force constant
        # based on default time step
        if not checkup:
            c = (idt * (f - fr * v) / 2
                 + sqrt_idt * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_3) / 4)
            d = sqrt_idt * sig * eta / (2 * sqrt_3)
        else:
            # if checkup is True, this means we already performed an entire
            # propagation cycle and have already updated the velocities
            # based on values for the full default time step. thus we
            # need to make sure not to update velocities a second time
            # because that would destroy energy conservation.
            c = np.asarray([np.asarray([0., 0., 0.]) for atom in atoms])
            d = np.asarray([np.asarray([0., 0., 0.]) for atom in atoms])

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
                v = (self.atoms.get_positions() - x - dt * d) / dt
                atoms.set_momenta(v * m)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)
                v = (self.atoms.get_positions() - x - dt * d) / dt
                atoms.set_momenta(v * m, apply_constraint=False)

        return atoms

    def predictConflicts(self, atoms, forces, dt, halfstep, 
                         constraints, checkup):
        """Test-propagate atoms and check if boundary conflict occurs.

        Parameters:

        atoms: object
            ASE atoms object holding the current configuration.

        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        dt: float
            The time step used to propagate.

        halfstep: int (1 or 2)
            Boundary conflict resolution is performed in two halfsteps.
            1: propagate by fraction of dt so that outermost inner
               particle and innermost outer particle have the same
               distance from the solute, redirect particles.
            2: after conflict resolution, propagate remaining
               fraction of dt to complete a full default time step.

        constraints: bool
            Defaults to True, turn constraints on or off during the
            propagation.

        checkup: bool
            False for first conflict resolution within a given time
            step, True for any subsequent conflict resolitions within
            the same time step.
        """

        # Propagate a copy of the atoms object by self.dt.
        FTatoms = atoms.copy()
        future_atoms = self.propagate(FTatoms, forces, dt, 
                checkup=checkup, halfstep=halfstep, constraints=constraints)

        # Convert future_atoms to COM atoms objects.
        (ftr_com_atoms, ftr_com_forces, ftr_r, ftr_d,
        ftr_boundary_idx, ftr_boundary) = self.update(future_atoms, forces)

        # Check if future_atoms contains boundary conflict.
        conflicts = [(ftr_boundary_idx, atom.index) for atom in ftr_com_atoms
                     if atom.tag == 3 and ftr_d[atom.index] < ftr_boundary]

        return conflicts

    def collide(self, atoms, forces, inner_reflect, outer_reflect):
        """Perform elastic collision between two paticles.

        Parameters:

        atoms: object
            ASE atoms object holding the current configuration.

        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        inner_reflect, outer_reflect: int
            Atoms object indices of the COM-reduced pseudoparticles
            that the collision is supposed to be performed with.
        """
        
        # Update parameters.
        com_atoms, com_forces, r, d, boundary_idx, boundary = (
                self.update(atoms, forces))
        r_inner = r[inner_reflect]
        r_outer = r[outer_reflect]
        m_outer = com_atoms[outer_reflect].mass
        m_inner = com_atoms[inner_reflect].mass
        v_outer = com_atoms[outer_reflect].momentum / m_outer
        v_inner = com_atoms[inner_reflect].momentum / m_inner

        # Find angle between r_outer and r_inner.
        theta = self.calc_angle(r_inner, r_outer)
        self.debuglog("   angle (r_outer, r_inner) is: {:.16f}\n"
                      .format(np.degrees(theta)))

        # Rotate OUTER to be exactly on top of the INNER for
        # collision. this simulates the boundary mediating
        # a collision between the particles.

        # calculate rotational axis
        axis = self.normalize(np.cross(r[outer_reflect],
                              r[inner_reflect]))

        # rotate velocity of outer region particle
        v_outer = np.dot(self.rotation_matrix(axis, theta),
                         v_outer)

        # Perform mass-weighted exchange of normal components of
        # velocitiy, force (, and random forces if Langevin).
        # i.e. elastic collision
        if self.reflective:
            self.debuglog("   -> hard wall reflection\n")
            n = self.normalize(r_inner)
            if np.dot(v_inner, n) > 0:
                dV_inner = -2 * np.dot(np.dot(v_inner, n), n)
            else:
                dV_inner = np.array([0., 0., 0.])
            if np.dot(v_outer, n) < 0:
                dV_outer = -2 * np.dot(np.dot(v_outer, n), n)
            else:
                dV_outer = np.array([0., 0., 0.])
            self.debuglog("   dV_inner = {:s}\n"
                          .format(np.array2string(dV_inner)))
            self.debuglog("   dV_outer = {:s}\n"
                          .format(np.array2string(dV_outer)))
        else:
            self.debuglog("   -> momentum exchange collision\n")
            M = m_outer + m_inner
            r12 = r_inner
            v12 = v_outer - v_inner
            self.debuglog("   r12 = {:s}\n"
                          .format(np.array2string(r12)))
            self.debuglog("   v12 = {:s}\n"
                          .format(np.array2string(v12)))
            self.debuglog("   dot(v12, r12) = {:.16f}\n"
                          .format(np.dot(v12, r12)))
            v_norm = np.dot(v12, r12) * r12 / (np.linalg.norm(r12)**2)
            # dV_inner and dV_outer should be mass weighted differently
            # to allow for momentum exchange between solvent with diff
            # total masses.
            dV_inner = 2 * m_inner / M * v_norm
            self.debuglog("   dV_inner = {:s}\n"
                          .format(np.array2string(dV_inner)))
            dV_outer = -2 * m_outer / M * v_norm
            self.debuglog("   dV_outer = {:s}\n"
                          .format(np.array2string(dV_outer)))

        if not self.surface and theta != 0 and theta != np.pi:
            # rotate outer particle velocity change component
            # back to inital direction after collision
            dV_outer = np.dot(self.rotation_matrix(
                              axis, -1 * theta), dV_outer)

        if theta == np.pi:
            # flip velocity change component of outer particle
            # to the other side of the slab (if applicable)
            dV_outer = -1 * dV_outer

        # commit new momenta to pseudoparticle atoms object
        com_atoms[outer_reflect].momentum += (dV_outer * m_outer)
        com_atoms[inner_reflect].momentum += (dV_inner * m_inner)

        # expand the pseudoparticle atoms object back into the
        # real atoms object (inverse action to self.update())
        outer_actual = self.idx_real[outer_reflect]
        inner_actual = self.idx_real[inner_reflect]

        mom = atoms.get_momenta()
        m = self.masses

        mom[outer_actual:outer_actual+self.nout] += np.tile(dV_outer, 
                (self.nout, 1)) * m[outer_actual:outer_actual+self.nout]

        mom[inner_actual:inner_actual+self.nin] += np.tile(dV_inner, 
                (self.nin, 1)) * m[inner_actual:inner_actual+self.nin]
        
        atoms.set_momenta(mom, apply_constraint=False)

        # keep track of which pair of conflicting particles
        # was just resolved for future reference
        self.recent = [inner_reflect, outer_reflect]

        return atoms

    def step(self, forces=None):
        """Perform a SAFIRES MD step."""

        atoms = self.atoms
        lenatoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        # Must propagate future atoms object with non-modified force array 
        # to predict the correct motion of the COM
        NCforces = atoms.get_forces(md=False)

        # This velocity as well as xi, eta and a few other variables are stored
        # as attributes, so Asap can do its magic when atoms migrate between
        # processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(lenatoms, 3))
        self.eta = self.rng.standard_normal(size=(lenatoms, 3))

        # To keep the center of mass stationary, the random arrays should add to (0,0,0)
        if self.fix_com:
            # Hack to make this BS work with FixAtoms (for now)
            for i, c in enumerate(atoms.constraints):
                if c.todict()['name'] in _allowed_constraints:
                    self.xi[c.index]  = 0.0
                    self.eta[c.index] = 0.0
 
            self.xi -= self.xi.sum(axis=0) / lenatoms
            self.eta -= self.eta.sum(axis=0) / lenatoms
 
            # We run this again because of obvious reasons
            # Hack to make this BS work with FixAtoms (for now)                     
            for i, c in enumerate(atoms.constraints):
                if c.todict()['name'] in _allowed_constraints:
                    self.xi[c.index]  = 0.0
                    self.eta[c.index] = 0.0

        self.communicator.broadcast(self.xi, 0)
        self.communicator.broadcast(self.eta, 0)

        # The checkup bool is used to differentiate between the first
        # and any potential subsequent collision resolutions as they
        # have to be treated differently.
        checkup = False

        # Predict boundary conflicts after propagating by self.dt.
        conflicts = self.predictConflicts(atoms=atoms, 
                                          forces=NCforces, 
                                          dt=self.dt,
                                          halfstep=1,
                                          constraints=False,
                                          checkup=checkup)

        # If there are boundary conflicts, execute problem solving.
        if conflicts:
            while conflicts:
                print("CONFLICTS = ", conflicts)
                print("Current iteration = ", self.get_number_of_steps())
                print("regular dt = ", self.dt)
                
                # Find the conflict that occurs earliest in time
                dt_list = [self.extrapolate_dt(atoms, NCforces, c[0], c[1], 
                           checkup=False) for c in conflicts]
                conflict = sorted(dt_list, key=itemgetter(2))[0]
                print("First conflict = ", conflict)

                # When holonomic constraints for rigid linear triatomic molecules are
                # present, ask the constraints to redistribute xi and eta within each
                # triple defined in the constraints. This is needed to achieve the
                # correct target temperature.
                for constraint in atoms.constraints:
                    if hasattr(constraint, 'redistribute_forces_md'):
                        constraint.redistribute_forces_md(atoms, self.xi, rand=True)
                        constraint.redistribute_forces_md(atoms, self.eta, rand=True)

                print("d before boundary propagation")
                xx, xx, xx, d, xx, xx = (
                    self.update(atoms, forces))
                print("d_inner = ", d[conflict[0]])
                print("d_outer = ", d[conflict[1]])

                # Propagate to boundary.
                atoms = self.propagate(atoms, forces, conflict[2], checkup=checkup,
                            halfstep=1, constraints=True)
                
                print("d after boundary propagation")
                xx, xx, xx, d, xx, xx = (
                    self.update(atoms, forces))
                print("d_inner = ", d[conflict[0]])
                print("d_outer = ", d[conflict[1]])

                # Resolve elastic collision.
                atoms = self.collide(atoms, forces, conflict[0], conflict[1])
                
                print("d after collision")
                xx, xx, xx, d, xx, xx = (
                    self.update(atoms, forces))
                print("d_inner = ", d[conflict[0]])
                print("d_outer = ", d[conflict[1]])

                # Propagate remaining time step.
                if self.remaining_dt == 0:
                    self.remaining_dt = self.dt - conflict[2]
                else:
                    self.remaining_dt -= conflict[2]

                # Update checkup. 
                checkup = True
            
                # Predict boundary conflicts after propagating by 
                # self.remaining_dt.
                conflicts = self.predictConflicts(atoms=atoms, 
                                                  forces=NCforces, 
                                                  dt=self.remaining_dt,
                                                  halfstep=1,
                                                  constraints=False,
                                                  checkup=checkup)
        
            # After all conflicts are resolved, the second propagation
            # halfstep is executed.
            print("Remaining_dt after collision = ", self.remaining_dt)
            atoms = self.propagate(atoms, forces, self.remaining_dt,
                        checkup=checkup, halfstep=2, constraints=True)
            
            print("d after makeup propagation")
            xx, xx, xx, d, xx, xx = (
                self.update(atoms, forces))
            print("d_inner = ", d[conflict[0]])
            print("d_outer = ", d[conflict[1]])

        # No conflict: regular propagation
        else:
            for constraint in atoms.constraints:
                if hasattr(constraint, 'redistribute_forces_md'):
                    constraint.redistribute_forces_md(atoms, self.xi, rand=True)
                    constraint.redistribute_forces_md(atoms, self.eta, rand=True)

            atoms = self.propagate(atoms, forces, self.dt, checkup=checkup,
                halfstep=1, constraints=True)

        # Finish propagation as usual.
        m = self.masses
        v = atoms.get_velocities()
        T = self.temp
        fr = self.fr
        xi = self.xi
        eta = self.eta
        sig = np.sqrt(2 * T * fr / m)

        forces = atoms.get_forces(md=True)
        f = forces / m
        c = (self.dt * (f - fr * v) / 2
             + math.sqrt(self.dt) * sig * xi / 2
             - self.dt * self.dt * fr * (f - fr * v) / 8
             - self.dt**1.5 * fr * sig * (xi / 2 + eta / math.sqrt(3)) / 4)
        v += c
        atoms.set_momenta(v * m)

        # Unset tracking variables.
        self.remaining_dt = 0
        self.recent = []

        return forces
