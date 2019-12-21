"""Bussi NVT dynamics class."""

import math
import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world


def sumnoises(nn):
    """Sum of nn noises."""
    if nn == 0:
        return 0.
    elif nn % 2 == 0:
        return 2 * np.random.gamma(nn / 2, 1.)
    else:
        rr = np.random.normal(scale=1.)
        return 2 * np.random.gamma((nn - 1) / 2, 1.) + rr**2


def resamplekin(kk, sigma, ndeg, taut):
    """Resample the kinetic energy.

    Args:
        kk
            present value of the kinetic energy of the atoms to be thermalized
            (in arbitrary units)
        sigma
            target average value of the kinetic energy (ndeg k_b T/2)
            (in the same units as kk)
        ndeg
            number of degrees of freedom of the atoms to be thermalized
        taut
            relaxation time of the thermostat, in units of 'how often this
            routine is called'
    """
    if taut > 0.1:
        factor = math.exp(-1. / taut)
    else:
        factor = 0.
    rr = np.random.normal(scale=1.)
    return (
        kk +
        (1. - factor) * (sigma * (sumnoises(ndeg - 1) + rr**2) / ndeg - kk) +
        2. * rr * math.sqrt(kk * sigma / ndeg * (1. - factor) * factor)
    )


class Bussi(MolecularDynamics):
    """Bussi stochastic velocity rescaling (NVT) molecular dynamics.

    Usage: Bussi(atoms, timestep, temperature, taut, fixcm)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        The desired temperature, in Kelvin.

    taut
        Time constant for Bussi temperature coupling.

    """

    def __init__(self, atoms, timestep, temperature, taut, fixcm=True,
                 trajectory=None, logfile=None, loginterval=1,
                 communicator=world, append_trajectory=False):

        MolecularDynamics.__init__(self, atoms, timestep, trajectory,
                                   logfile, loginterval,
                                   append_trajectory=append_trajectory)
        self.taut = taut
        self.temperature = temperature
        self.fixcm = fixcm
        self.communicator = communicator

    def set_taut(self, taut):
        self.taut = taut

    def get_taut(self):
        return self.taut

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_temperature(self):
        return self.temperature

    def set_timestep(self, timestep):
        self.dt = timestep

    def get_timestep(self):
        return self.dt

    def scale_velocities(self):
        """ Do the NVT Bussi stochastic velocity scaling """
        kenergy = self.atoms.get_kinetic_energy()
        # initialize velocities if kinetic energy is zero; the scaling factor
        # would be infinite otherwise
        if kenergy < 1e-12:
            self.atoms.set_velocities(
                1e-3 * np.random.random((len(self.atoms), 3)))
            kenergy = self.atoms.get_kinetic_energy()
        ndims = 3  # ?
        ndeg = len(self.atoms) * ndims - ndims

        new_kenergy = resamplekin(
            kenergy, 0.5 * self.temperature * ndeg, ndeg, self.taut / self.dt)
        scl_temperature = math.sqrt(new_kenergy / kenergy)
        # print("Bussi old", kenergy, "new", new_kenergy,
        #       "scaling temperature by", scl_temperature)

        p = self.atoms.get_momenta()
        p = scl_temperature * p
        self.atoms.set_momenta(p)

    def step(self, f=None):
        """Move one timestep forward using Bussi NVT molecular dynamics."""
        self.scale_velocities()

        # one step velocity verlet
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * f

        if self.fixcm:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum

        self.atoms.set_positions(
            self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:, np.newaxis])

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        self.atoms.set_momenta(p)
        f = self.atoms.get_forces()
        atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * f)

        return f
