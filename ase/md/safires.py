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
from ase.parallel import parprint

_allowed_constraints = {'FixAtoms', 'FixCom'}


class SAFIRES(MolecularDynamics):
    """SAFIRES (constant N, V, T, variable dt) molecular dynamics."""

    def __init__(self, atoms, timestep, temperature=None, friction=None,
                 natoms=None, natoms_in=None, fixcm=True, *, 
                 temperature_K=None, trajectory=None, logfile=None, 
                 loginterval=1, communicator=world, rng=None, 
                 append_trajectory=False, surface=False, reflective=False,
                 debug=False):
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

        debug: bool (optional)
            Defaults to False, will prompt SAFIRES to create an additional
            debug output file named "safires_debug.log" with verbose debug
            output. Please activate and attach this file when reporting bugs.

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
        
        # SAFIRES specific setups.
        self.reflective = reflective
        self.surface = surface
        self.constraints = atoms.constraints.copy()
        self.recent = []
        self.remaining_dt = 0
        self.current_boundary = 0.
        self.nconflicts = 0
        self.debug = debug
        
        # SAFIRES: determine number of atoms of inner and
        # outer region solvent molecules.
        self.nout = natoms
        if natoms_in is not None:
            self.nin = natoms_in
        else:
            self.nin = natoms
        # Relative index of pseudoparticle in total atoms object.
        self.idx_real = None
        self.nsol = len([atom.index for atom in atoms
                       if atom.tag == 1])
        # SAFIRES: Set up natoms array according to tag : 0, 1, 2, 3.
        self.nall = np.array([1, self.nsol, self.nin, self.nout])

        # SAFIRES: assertions / warnings to ensure functionality.
        assert any(4 > atom.tag for atom in atoms), \
                   'Atom tag 4 and above not supported'
        assert any(1 == atom.tag for atom in atoms), \
                   'Solute is not tagged (tag=1) correctly'
        assert any(2 == atom.tag for atom in atoms), \
                   'Inner region is not tagged (tag=2) correctly'
        assert any(3 == atom.tag for atom in atoms), \
                   'Outer region is not tagged (tag=3) correctly'

        # SAFIRES: warn if masses of inner and outer solvent are different.
        m_out = np.array([atom.mass for atom in atoms if atom.tag == 3])
        nm_out = int(len(m_out) / self.nout)
        m_in = np.array([atom.mass for atom in atoms if atom.tag == 2])
        nm_in = int(len(m_in) / self.nin)
        if not (np.round(m_out.sum() / nm_out, decimals=10) == np.round(
                m_in.sum() / nm_in, decimals=10)):
            warnings.warn('The mass of inner and outer solvent molecules is \
                           not exactly the same')

        # SAFIRES: solute must be constrained in a specific way.
        assert atoms.constraints, \
               'Constraints are not set (solute can not move)'
        assert self.check_constraints(atoms), \
               'Solute constraint not correctly set'

        # Final sanity check, make sure all inner are closer to 
        # origin than outer.
        assert self.check_distances(atoms), \
               'Outer molecule closer to origin than inner'

        # Open file for the SAFIRES-specific debug logger function.
        if self.debug:
            self.debuglog = open("debug_safires.log", "w+")
            self.writeDebug("SAFIRES DEBUG OUTPUT\n")
            self.writeDebug("====================\n\n")
            self.writeDebug("Regular timestep: {:f}\n".format(timestep))
            self.writeDebug("Temperature: {:f}\n".format(self.temp))
            self.writeDebug("Friction: {:f}\n\n".format(self.fr))
            tag0 = np.array([atom.index for atom in atoms if atom.tag == 0])
            tag1 = np.array([atom.index for atom in atoms if atom.tag == 1])
            tag2 = np.array([atom.index for atom in atoms if atom.tag == 2])
            tag3 = np.array([atom.index for atom in atoms if atom.tag == 3])
            self.writeDebug("Atoms tag == 0 (ignore): {:s} (len: {:d})\n"
                                .format(np.array2string(tag0), len(tag0)))
            self.writeDebug("Atoms tag == 1 (solute): {:s} (len: {:d})\n"
                                .format(np.array2string(tag1), len(tag1)))
            self.writeDebug("Atoms tag == 2 (inner): {:s} (len: {:d})\n"
                                .format(np.array2string(tag2), len(tag2)))
            self.writeDebug("Atoms tag == 3 (outer): {:s} (len: {:d})\n"
                                .format(np.array2string(tag3), len(tag3)))
            self.writeDebug("Assuming that inner molecules have"
                            " {:d} atoms each\n".format(self.nin))
            self.writeDebug("Assuming that outer molecules have"
                            " {:d} atoms each\n".format(self.nout))
            self.writeDebug("\nSpecified SAFIRES behaviors:\n")
            if natoms > 1:
                self.writeDebug("- natoms is > 1; note that all output in"
                                " this output file refers to COM-reduced"
                                " pseudoparticles\n")
            if self.reflective:
                self.writeDebug("- SAFIRES in reflective mode\n")
            else:
                self.writeDebug("- SAFIRES in elastic collision mode\n")
            if self.surface:
                self.writeDebug("- SAFIRES assumes a periodic surface"
                                " as the solute\n")
            else:
                self.writeDebug("- SAFIRES assumes a molecular solute\n")
            self.writeDebug("\n=> STARTING SAFIRES MOLECULAR DYNAMICS\n")

        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)

    def __del__(self):
        if self.debug:
            self.debuglog.close()
    
    def writeDebug(self, content):
        self.debuglog.write(content)
        self.debuglog.flush()

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
        if self.surface:
            cm_origin[0] = 0.
            cm_origin[1] = 1.

        cm_inner = []
        cm_outer = []
        
        i = 0
        while i < len(atoms):
            tag = atoms[i].tag
            idx = atoms[i].index
            nat = self.nall[tag]
            if tag == 2:
                tmp = atoms[idx:idx + nat].get_center_of_mass()
                if self.surface:
                    tmp[0] = 0.
                    tmp[1] = 0.
                cm_inner.append(tmp)
            if tag == 3:
                tmp = atoms[idx:idx + nat].get_center_of_mass()
                if self.surface:
                    tmp[0] = 0.
                    tmp[1] = 0.
                cm_outer.append(tmp)
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

    def get_boundary(self):
        return self.current_boundary

    def get_number_of_conflicts(self):
        return self.nconflicts

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

    def update(self, atoms, forces, update_boundary=False):
        """Return reduced pseudoparticle atoms object.

        Parameters:
        
        atoms: object
            ASE atoms object holding current atomic configuration.
        
        forces: array
            Numpy array containing the vectorfield of forces on
            all atoms.

        update_boundary: bool (optional)
            Defaults to False, updates self.current_boundary
            which is read by MDLogger.

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
        com_atoms = Atoms()
        com_atoms.pbc = atoms.pbc
        com_atoms.cell = atoms.cell

        # Get_center_of_mass() breaks with certain constraints.
        self.constraints = atoms.constraints.copy()
        atoms.constraints = []

        # Calculate cumulative properties for all inner/outer
        # region particles. For monoatomic inner/outer particles,
        # a 1:1 copy is created.
        com_forces = []
        idx_real = []
        i = 0
        while i < len(atoms):
            # Calculate cumulative properties.
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

            # Create new atoms.
            tmp = Atoms(sym)
            tmp.set_positions([com])
            tmp.set_momenta([mom])
            tmp.set_masses([M])
            tmp.set_tags([tag])
            com_forces.append(frc)

            # Append and iterate.
            com_atoms += tmp
            idx_real.append(i)
            i += nat

        if self.surface:
            # Only need z coordinates for surface calculations.
            for atom in com_atoms:
                atom.position[0] = 0.
                atom.position[1] = 0.
        
        # idx_real is used to re-expand the reduced pseuodparticle
        # atoms object into the original atoms object.
        self.idx_real = idx_real
        
        # Reapply constraints to original atoms object.
        atoms.constraints = self.constraints.copy()

        # Calculate distances between COM of solute and all inner and
        # outer region particles.
        r, d = find_mic([atom.position for atom in com_atoms] - sol_com,
                       com_atoms.cell, com_atoms.pbc)

        # List all particles in the inner region.
        inner_mols = [(atom.index, d[atom.index])
                      for atom in com_atoms if atom.tag == 2]

        # Boundary is defined by the inner region particle that has
        # the largest absolute distance from the COM of the solute.
        boundary_idx, boundary = sorted(inner_mols, key=itemgetter(1),
                                        reverse=True)[0]
        if update_boundary:
            self.current_boundary = boundary

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
        # Results dictionary. 
        # Structure: (inner particle index, extrapolated factor)
        res = []
            
        # Extrapolation uses reduced COM atoms object.
        com_atoms, com_forces, r, d, boundary_idx, boundary = (
                self.update(atoms, forces))

        # Eliminate duplicate entries, then extraplate time step
        # required to propagate inner and outer particle to the
        # same distance from the solute.
        for inner_idx in set([boundary_idx, ftr_boundary_idx]):
            # If multiple conflict resolutions take place in one time
            # step, ignore already solved conflicts.
            if inner_idx in self.recent and outer_idx in self.recent:
                continue

            # Retreive necessary ingredients for extrapolation.
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

            # Re-expand original atom object indices of involved molecules.
            outer_real = self.idx_real[outer_idx]
            inner_real = self.idx_real[inner_idx]

            # If inner/outer particles are molecules:
            if self.nout > 1 or self.nin > 1:
                # Calculate cumulative properties on each molecule.
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

                # TODO: check here if fr is array, expand accordingly.
                sig_outer = math.sqrt(2 * T * fr)
                sig_inner = math.sqrt(2 * T * fr)
                
            # If inner/outer particles are monoatomic:
            else:
                xi_outer = self.xi[outer_real]
                xi_inner = self.xi[inner_real]
                eta_outer = self.eta[outer_real]
                eta_inner = self.eta[inner_real]
                sig_outer = math.sqrt(2 * T * fr / m_outer)
                sig_inner = math.sqrt(2 * T * fr / m_inner)

            # Surface calculations: only z components required.
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

            # Solve 2nd degree polynomial of the form:
            # y = c0*x^2 + c1*x + c2.
            # a and b are velocity modifiers derived from the first
            # velocity half step in the Langevin algorithm.
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
            
            else:
                a_outer = 0
                a_inner = 0
                b_outer = 0
                b_inner = 0

            # Update velocities.
            v_outer += a_outer
            v_outer += b_outer
            v_inner += a_inner
            v_inner += b_inner

            # Set up polynomial coefficients.
            c0 = np.dot(r_inner, r_inner) - np.dot(r_outer, r_outer)
            c1 = 2 * np.dot(r_inner, v_inner) - 2 * np.dot(r_outer, v_outer)
            c2 = np.dot(v_inner, v_inner) - np.dot(v_outer, v_outer)

            # Solve polynomial for roots.
            roots = np.roots([c2, c1, c0])
            if self.debug:
                self.writeDebug("   1) TIME STEP EXTRAPOLATION\n"
                          "      all extrapolated roots: {:s}\n"
                          .format(np.array2string(roots)))

            for val in roots:
                if np.isreal(val) and val <= self.dt and val > 0:
                    # The quadratic polynomial yields four roots.
                    # The smallest positive real value is the 
                    # desired time step.
                    res.append((inner_idx, outer_idx, np.real(val)))

        if not res:
            # Break if none of the obtained roots fit the criteria.
            # -> No way for SAFIRES to continue.
            error = ("\nERROR:\n"
                     "Unable to extrapolate time step.\n"
                     "All real roots <= 0 or > default time step).\n"
                     "Roots: {:s}\n".format(np.array2string(roots)))
            if self.debug:
                self.writeDebug(error)
            raise SystemExit(error)
        else:
            # Return desired timestep.
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
        # Retreive parameters.
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

        # Pre-calculate (random) force constant
        # based on default time step.
        if not checkup:
            c = (idt * (f - fr * v) / 2
                 + sqrt_idt * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_3) / 4)
            d = sqrt_idt * sig * eta / (2 * sqrt_3)
        else:
            # Don't update velocities again on additional conflict
            # resolution cycles after the first one in this time step.
            c = np.asarray([np.asarray([0., 0., 0.]) for atom in atoms])
            d = np.asarray([np.asarray([0., 0., 0.]) for atom in atoms])

        if halfstep == 1:
            # Friction and (random) forces should only be applied 
            # during the first halfstep.
            v += c + d
            if constraints:
                atoms.set_positions(x + dt * v)
                v = (atoms.get_positions() - x) / dt
                atoms.set_momenta(v * m)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)
                v = (atoms.get_positions() - x) / dt
                atoms.set_momenta(v * m, apply_constraint=False)

        if halfstep == 2:
            # Velocity update occurs at the end of step(),
            # after force update. Halfstep = 2 is only triggered during
            # iterations where a conflict was resolved.
            if constraints:
                atoms.set_positions(x + dt * v)
                v = (self.atoms.get_positions() - x - dt * d) / dt
                atoms.set_momenta(v * m)
            else:
                atoms.set_positions(x + dt * v, apply_constraint=False)
                v = (self.atoms.get_positions() - x - dt * d) / dt
                atoms.set_momenta(v * m, apply_constraint=False)

        return atoms

    def predictConflicts(self, atoms, forces, dt, halfstep, 
                    constraints, checkup, update_boundary=False):
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

        update_boundary: bool (optional)
            Defaults to False, flag used to communicate to self.update
            to update the self.current_boundary value which is read
            and logged by the MDLogger class.
        """
        # Propagate a copy of the atoms object by self.dt.
        FTatoms = atoms.copy()
        future_atoms = self.propagate(FTatoms, forces, dt, 
                checkup=checkup, halfstep=halfstep, constraints=constraints)

        # Convert future_atoms to COM atoms objects.
        (ftr_com_atoms, ftr_com_forces, ftr_r, ftr_d,
        ftr_boundary_idx, ftr_boundary) = self.update(future_atoms, forces, 
                                            update_boundary=update_boundary)

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
        # Fetch required ingrendients.
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
        if self.debug:
            self.writeDebug("   2) COLLISION\n"
                            "      angle (r_outer, r_inner) is: {:.16f}\n"
                             .format(np.degrees(theta)))

        # Rotate r_outer to be exactly on top of the r_inner.
        # This simulates the boundary mediating a collision between
        # the particles.
        axis = self.normalize(np.cross(r[outer_reflect],
                              r[inner_reflect]))
        v_outer = np.dot(self.rotation_matrix(axis, theta),
                         v_outer)

        # Perform mass-weighted exchange of normal components of
        # velocities, or hard wall collision (reflective = True).
        if self.reflective:
            n = self.normalize(r_inner)
            if np.dot(v_inner, n) > 0:
                dV_inner = -2 * np.dot(np.dot(v_inner, n), n)
            else:
                dV_inner = np.array([0., 0., 0.])
            if np.dot(v_outer, n) < 0:
                dV_outer = -2 * np.dot(np.dot(v_outer, n), n)
            else:
                dV_outer = np.array([0., 0., 0.])
        else:
            M = m_outer + m_inner
            r12 = r_inner
            v12 = v_outer - v_inner
            if self.debug:
                self.writeDebug("      r12 = {:s}\n"
                              .format(np.array2string(r12)))
                self.writeDebug("      v12 = {:s}\n"
                              .format(np.array2string(v12)))
                self.writeDebug("      dot(v12, r12) = {:.16f}\n"
                              .format(np.dot(v12, r12)))
            v_norm = np.dot(v12, r12) * r12 / (np.linalg.norm(r12)**2)
            
            # dV_inner and dV_outer are mass weighted differently
            # to allow for momentum exchange between solvent with diff
            # total masses.
            dV_inner = 2 * m_inner / M * v_norm
            dV_outer = -2 * m_outer / M * v_norm

        if self.debug:
            self.writeDebug("      dV_inner = {:s}\n"
                          .format(np.array2string(dV_inner)))
            self.writeDebug("      dV_outer = {:s}\n"
                          .format(np.array2string(dV_outer)))

        if not self.surface and theta != 0 and theta != np.pi:
            # Rotate outer particle velocity change component
            # back to inital direction.
            dV_outer = np.dot(self.rotation_matrix(
                              axis, -1 * theta), dV_outer)

        if theta == np.pi:
            # Flip velocity change component of outer particle
            # to the other side of a symmetric slab model.
            dV_outer = -1 * dV_outer

        # Commit new momenta to pseudoparticle atoms object.
        com_atoms[outer_reflect].momentum += (dV_outer * m_outer)
        com_atoms[inner_reflect].momentum += (dV_inner * m_inner)

        # Expand the pseudoparticle atoms object back into the
        # original atoms object (inverse action to self.update()).
        outer_actual = self.idx_real[outer_reflect]
        inner_actual = self.idx_real[inner_reflect]
        mom = atoms.get_momenta()
        m = self.masses
        mom[outer_actual:outer_actual+self.nout] += np.tile(dV_outer, 
                (self.nout, 1)) * m[outer_actual:outer_actual+self.nout]

        mom[inner_actual:inner_actual+self.nin] += np.tile(dV_inner, 
                (self.nin, 1)) * m[inner_actual:inner_actual+self.nin]
        atoms.set_momenta(mom, apply_constraint=False)

        # Keep track of which pair of conflicting particles
        # was just resolved for future reference.
        self.recent.extend([inner_reflect, outer_reflect])

        return atoms

    def step(self, forces=None):
        """Perform a SAFIRES MD step."""
        atoms = self.atoms
        lenatoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        if self.debug:
            self.writeDebug("\nIteration: {:d}\n".format(
                            self.get_number_of_steps()))

        # Must propagate future atoms object with non-modified force
        # array to predict the correct motion of the COM.
        NCforces = atoms.get_forces(md=False)

        # This velocity as well as xi, eta and a few other variables
        # are stored as attributes, so Asap can do its magic when
        # atoms migrate between processors.
        self.v = atoms.get_velocities()

        self.xi = self.rng.standard_normal(size=(lenatoms, 3))
        self.eta = self.rng.standard_normal(size=(lenatoms, 3))

        # To keep the center of mass stationary, the random arrays
        # should add to (0,0,0).
        if self.fix_com:
            # Hack to make this SAFIRES work with FixAtoms.
            # TODO: find a more holistic solution.
            for i, c in enumerate(atoms.constraints):
                if c.todict()['name'] in _allowed_constraints:
                    self.xi[c.index] = 0.0
                    self.eta[c.index] = 0.0
            idxChg = [atom.index for atom in atoms if atom.tag != 1]
            lenChg = len(idxChg)
            self.xi[idxChg] -= self.xi[idxChg].sum(axis=0) / lenChg
            self.eta[idxChg] -= self.eta[idxChg].sum(axis=0) / lenChg
 
        # Run again to remove any random forces potentially
        # redistributed to solute.
        # TODO: find a more holistic solution.
        for i, c in enumerate(atoms.constraints):
            if c.todict()['name'] in _allowed_constraints:
                self.xi[c.index] = 0.0
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
                                          checkup=checkup,
                                          update_boundary=True)

        # If there are boundary conflicts, execute problem solving.
        if conflicts:
            while conflicts:
                self.nconflicts += 1
                if self.debug:
                    self.writeDebug("   CONFLICT DETECTED (total: {:d})\n"
                                    .format(self.nconflicts))
                
                # Find the conflict that occurs earliest in time.
                dt_list = [self.extrapolate_dt(atoms, NCforces, c[0], c[1], 
                           checkup=False) for c in conflicts]
                conflict = sorted(dt_list, key=itemgetter(2))[0]
                if self.debug:
                    self.writeDebug("      Treating conflict [a1, a2, dt]:"
                            " [{:d}, {:d}, {:.16f}]\n".format(conflict[0],
                            conflict[1], conflict[2]))
                # Write event info to stdout.
                parprint("".join(["<SAFIRES> Iteration {:d}: "
                           .format(self.get_number_of_steps()),
                           "Treating atoms {:d} and {:d} at d = {:.5f}"
                           .format(conflict[0], conflict[1], 
                                   self.current_boundary),
                           " using dt = {:.5f}"
                           .format(conflict[2])]))

                # When holonomic constraints for rigid linear triatomic
                # molecules are present, ask the constraints to 
                # redistribute xi and eta within each triple defined in
                # the constraints. This is needed to achieve the 
                # correct target temperature.
                for constraint in atoms.constraints:
                    if hasattr(constraint, 'redistribute_forces_md'):
                        constraint.redistribute_forces_md(atoms, self.xi,
                                                          rand=True)
                        constraint.redistribute_forces_md(atoms, self.eta,
                                                          rand=True)
                if self.debug:
                    self.writeDebug("      d before boundary propagation:\n")
                    xx, xx, xx, d, xx, xx = (
                        self.update(atoms, forces))
                    self.writeDebug("         d_inner = {:.16f}\n".format(
                                    d[conflict[0]]))
                    self.writeDebug("         d_outer = {:.16f}\n".format(
                                    d[conflict[1]]))

                # Propagate to boundary.
                atoms = self.propagate(atoms, forces, conflict[2], 
                                       checkup=checkup, halfstep=1, 
                                       constraints=True)
                
                if self.debug:
                    self.writeDebug("      d after boundary propagation:\n")
                    xx, xx, xx, d, xx, xx = (
                        self.update(atoms, forces))
                    self.writeDebug("         d_inner = {:.16f}\n".format(
                                    d[conflict[0]]))
                    self.writeDebug("         d_outer = {:.16f}\n".format(
                                    d[conflict[1]]))

                # Resolve elastic collision.
                atoms = self.collide(atoms, forces, conflict[0], conflict[1])
                
                if self.debug:
                    self.writeDebug("      d after collision:\n")
                    xx, xx, xx, d, xx, xx = (
                        self.update(atoms, forces))
                    self.writeDebug("         d_inner = {:.16f}\n".format(
                                    d[conflict[0]]))
                    self.writeDebug("         d_outer = {:.16f}\n".format(
                                    d[conflict[1]]))

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
            atoms = self.propagate(atoms, forces, self.remaining_dt,
                        checkup=checkup, halfstep=2, constraints=True)
            
            if self.debug:
                self.writeDebug("      Remaining_dt after collision "
                                "= {:.16f}\n".format(self.remaining_dt))
                self.writeDebug("      d after makeup propagation:\n")
                xx, xx, xx, d, xx, xx = (
                    self.update(atoms, forces))
                self.writeDebug("         d_inner = {:.16f}\n".format(
                                d[conflict[0]]))
                self.writeDebug("         d_outer = {:.16f}\n".format(
                                d[conflict[1]]))

        # No conflict: regular propagation.
        else:
            if self.debug:
                self.writeDebug("   No conflict detected.\n")
            for constraint in atoms.constraints:
                if hasattr(constraint, 'redistribute_forces_md'):
                    constraint.redistribute_forces_md(atoms, self.xi, rand=True)
                    constraint.redistribute_forces_md(atoms, self.eta, rand=True)

            x = atoms.get_positions()
            m = self.masses
            v = atoms.get_velocities()
            T = self.temp
            fr = self.fr
            eta = self.eta
            sig = np.sqrt(2 * T * fr / m)
            d = self.dt**1.5 * sig * eta / (2 * math.sqrt(3))
            
            atoms = self.propagate(atoms, forces, self.dt, checkup=checkup,
                halfstep=1, constraints=True)
            # Recalc velocities after RATTLE constraints are applied.
            v = (self.atoms.get_positions() - x - d) / self.dt
            # Do not apply constraints here, this happens below after the
            # force update.
            atoms.set_momenta(v * self.masses, apply_constraint=False)

        # Finish propagation as usual (second velocity halfstep, 
        # update forces).
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

        # Reset tracking variables.
        self.remaining_dt = 0
        self.recent = []
        
        if self.debug:
            self.writeDebug("   Iteration concluded.\n")

        return forces
