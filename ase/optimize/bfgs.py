import warnings
from typing import IO, Optional, Union

import numpy as np
from numpy.linalg import eigh

from ase import Atoms
from ase.optimize.optimize import Optimizer


class BFGS(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 70.0}

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        master: Optional[bool] = None,
        alpha: Optional[float] = None,
    ):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn(
                'You are using a *very* large value for '
                'the maximum step size: %.1f Å' % self.maxstep
            )

        self.alpha = alpha
        if self.alpha is None:
            self.alpha = self.defaults['alpha']

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(3 * len(self.optimizable)) * self.alpha

        self.H = None
        self.pos0 = None
        self.forces0 = None

    def read(self):
        self.H, self.pos0, self.forces0, self.maxstep = self.load()

    def step(self, forces=None):
        optimizable = self.optimizable

        if forces is None:
            forces = optimizable.get_forces()

        pos = optimizable.get_positions()
        dpos, steplengths = self.prepare_step(pos, forces)
        dpos = self.determine_step(dpos, steplengths)
        optimizable.set_positions(pos + dpos)
        self.dump((self.H, self.pos0, self.forces0, self.maxstep))

    def prepare_step(self, pos, forces):
        forces = forces.reshape(-1)
        self.update(pos.flat, forces, self.pos0, self.forces0)
        omega, V = eigh(self.H)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dpos**2).sum(1) ** 0.5
        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()
        return dpos, steplengths

    def determine_step(self, dpos, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dpos *= scale
        return dpos

    def update(self, pos, forces, pos0, forces0):
        if self.H is None:
            self.H = self.H0
            return
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dforces = forces - forces0
        a = np.dot(dpos, dforces)
        dg = np.dot(self.H, dpos)
        b = np.dot(dpos, dg)
        self.H -= np.outer(dforces, dforces) / a + np.outer(dg, dg) / b

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory

            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        pos0 = atoms.get_positions().ravel()
        forces0 = atoms.get_forces().ravel()
        for atoms in traj:
            pos = atoms.get_positions().ravel()
            forces = atoms.get_forces().ravel()
            self.update(pos, forces, pos0, forces0)
            pos0 = pos
            forces0 = forces

        self.pos0 = pos0
        self.forces0 = forces0


class oldBFGS(BFGS):
    def determine_step(self, dpos, steplengths):
        """Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        """
        dpos /= np.maximum(steplengths / self.maxstep, 1.0).reshape(-1, 1)
        return dpos
