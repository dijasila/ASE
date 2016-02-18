import numpy as np

from ase.optimize.optimize import Optimizer
from ase.parallel import parprint


class FIRE(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxmove=0.2, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False):
        """Parameters:

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

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        downhill_check: boolean
            Check if energy actually decreased during FIRE steps, otherwise
            reset and decrease time step.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

        self.dt = dt
        self.Nsteps = 0
        self.maxmove = maxmove
        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.downhill_check = downhill_check

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()
       
    def step(self,f):
        atoms = self.atoms
        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
            if self.downhill_check:
                self.e_last = atoms.get_potential_energy()+1.0
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()
        else:
            is_uphill = False
            if self.downhill_check:
                e = atoms.get_potential_energy()
                # Check if the energy actually decreased
                if e > self.e_last:
                    # If not, reset to old positions...
                    parprint('Resetting positions because energy increased from'
                             ' %f to %f during step.' % (self.e_last, e))
                    atoms.set_positions(self.r_last)
                    is_uphill = True
                self.e_last = atoms.get_potential_energy()
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxmove:
            dr = self.maxmove * dr / normdr
        r = atoms.get_positions()
        atoms.set_positions(r + dr)
        self.dump((self.v, self.dt))
