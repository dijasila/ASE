from ase.optimize.optimize import Optimizer


class Internal(Optimizer):
    def __init__(self, atoms,
                 restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=None):
        super().__init__(atoms, restart, logfile, trajectory,
                         master, force_consistent=force_consistent)

    def run(self, fmax=0.05, steps=None):
        """call internal run and collect results"""
        self.atoms.calc.reset()

        self.fmax = fmax
        if steps:
            self.max_steps = steps

        self.atoms.calc.optimize(self.atoms, fmax, steps)
        self.atoms.set_positions(self.atoms.calc.atoms.get_positions(),
                                 apply_constraint=False)
