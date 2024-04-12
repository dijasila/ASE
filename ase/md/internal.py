from ase.io import Trajectory
from .md import MolecularDynamics


class Internal(MolecularDynamics):
    def __init__(self, atoms, timestep, temperature, trajectory=None):
        super().__init__(atoms, timestep, trajectory=None)
        self.temperature = temperature
        if isinstance(trajectory, str):
            trajectory = Trajectory(trajectory, 'a')
        self.trajectory = trajectory

    def run(self, steps=50):
        """ Call internal dynamics run"""
        self.atoms.calc.reset()

        self.atoms.calc.md(self.atoms,
                           self.dt,
                           self.temperature,
                           self.trajectory,
                           steps)

    def __del__(self):
        if self.trajectory is not None:
            self.trajectory.close()
