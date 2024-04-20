"""Saving the state of Dynamics objects to a file."""

from ase.io.jsonio import write_json
from ase.utils import IOContext


class RestartWriter(IOContext):

    def __init__(self, dyn, label):
        self.dyn = dyn
        self.label = label

    def __del__(self):
        self.close()

    def __call__(self):

        data = self.dyn.todict()
        name = f"{self.label}.restart.{self.dyn.nsteps}.json"

        with open(name, "w") as f:
            write_json(f, data)
