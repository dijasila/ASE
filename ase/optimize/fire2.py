# ######################################
# Implementation of FIRE2.0 and ABC-FIRE

# The FIRE2.0 algorithm is implemented using the integrator euler semi implicit
#  as described in the paper:
#   J. Guénolé, W.G. Nöhring, A. Vaid, F. Houllé, Z. Xie, A. Prakash,
#   E. Bitzek,
#    Assessment and optimization of the fast inertial relaxation engine (fire)
#    for energy minimization in atomistic simulations and its
#    implementation in lammps,
#    Comput. Mater. Sci. 175 (2020) 109584.
#    https://doi.org/10.1016/j.commatsci.2020.109584.
#    This implementation does not include N(p<0), initialdelay, dtmin
#
# ABC-Fire is implemented s described in the paper:
#   S. Echeverri Restrepo, P. Andric,
#    ABC-FIRE: Accelerated Bias-Corrected Fast Inertial Relaxation Engine,
#    Comput. Mater. Sci. 218 (2023) 111978.
#    https://doi.org/10.1016/j.commatsci.2022.111978.
#######################################

from typing import IO, Callable, Optional, Union

import numpy as np

from ase import Atoms
from ase.optimize.optimize import Optimizer


class FIRE2(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 20,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.25,
        fa: float = 0.99,
        master: Optional[bool] = None,
        position_reset_callback: Optional[Callable] = None,
        force_consistent=Optimizer._deprecated,
        abc: Optional[bool] = False
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        dt: float
            Initial time step. Defualt value is 0.1

        dtmax: float
            Maximum time step. Default value is 1.0

        dtmin: float
            Minimum time step. Default value is 2e-3

        finc: float
            Factor to increase the time step. Default value is 1.1

        fdec: float
            Factor to decrease the time step. Default value is 0.5

        astart: float
            Initial value of the parameter a. a is the Coefficcient for
            mixing the velocity and the force. Called alpha in the FIRE article.
            Default value 0.25.

        fa: float
            Factor to decrease the parameter alpha. Default value is 0.99

        Nmin: int
            Number of steps to wait after the last time the dot product of
            the velocity and force is negative (P in The FIRE article) before
            increasing the time step. Default value is 20.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2). Note that for ABC-FIRE the
            check is done independently for each cartesian direction.

        abc: bool
            If True, the Accelerated Bias-Corrected FIRE algorithm is
            used (ABC-FIRE).
            Default value is False.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        position_reset_callback: function(atoms, r, e, e_last)
            Function that takes current *atoms* object, an array of position
            *r* that the optimizer will revert to, current energy *e* and
            energy of last step *e_last*. This is only called if e > e_last.

        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)

        self.dt = dt

        self.Nsteps = 0

        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = self.defaults["maxstep"]

        self.dtmax = dtmax
        self.dtmin = dtmin
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = astart
        self.position_reset_callback = position_reset_callback
        self.abc = abc

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, f=None):
        optimizable = self.optimizable

        if f is None:
            f = optimizable.get_forces()

        if self.v is None:
            self.v = np.zeros((len(optimizable), 3))
        else:

            vf = np.vdot(f, self.v)
            if vf > 0.0:

                self.Nsteps += 1
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
            else:
                self.Nsteps = 0
                self.dt = max(self.dt*self.fdec, self.dtmin)
                self.a = self.astart

                dr = - 0.5 * self.dt * self.v
                r = optimizable.get_positions()
                optimizable.set_positions(r + dr)
                self.v[:] *= 0.0

        # euler semi implicit
        f = optimizable.get_forces()
        self.v += self.dt * f

        if self.abc:
            self.a = max(self.a, 1e-10)
            abc_multiplier = 1. / (1. - (1. - self.a)**(self.Nsteps + 1))
            v_mix = ((1.0 - self.a) * self.v + self.a * f / np.sqrt(
                np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v)))
            self.v = abc_multiplier * v_mix

            # Verifying if the maximum distance an atom
            #  moved is larger than maxstep, for ABC-FIRE the check
            #  is done independently for each cartesian direction
            if np.all(self.v):
                v_x = np.where(np.abs(self.v[:, 0]) * self.dt > self.maxstep,
                               (self.maxstep / self.dt) *
                               (self.v[:, 0] / np.abs(self.v[:, 0])),
                               self.v[:, 0])
                v_y = np.where(np.abs(self.v[:, 1]) * self.dt > self.maxstep,
                               (self.maxstep / self.dt) *
                               (self.v[:, 1] / np.abs(self.v[:, 1])),
                               self.v[:, 1])
                v_z = np.where(np.abs(self.v[:, 2]) * self.dt > self.maxstep,
                               (self.maxstep / self.dt) *
                               (self.v[:, 2] / np.abs(self.v[:, 2])),
                               self.v[:, 2])
                self.v = np.array([v_x, v_y, v_z]).T

        else:
            self.v = ((1.0 - self.a) * self.v + self.a * f / np.sqrt(
                np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v)))

        dr = self.dt * self.v

        # Verifying if the maximum distance an atom moved
        #  step is larger than maxstep, for FIRE2.
        if not self.abc:
            normdr = np.sqrt(np.vdot(dr, dr))
            if normdr > self.maxstep:
                dr = self.maxstep * dr / normdr

        r = optimizable.get_positions()
        optimizable.set_positions(r + dr)

        self.dump((self.v, self.dt))
