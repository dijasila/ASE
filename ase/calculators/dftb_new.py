import os
from copy import deepcopy
from ase.io import write
from ase.io.dftb import get_dftb_results
from ase.calculators.calculator import FileIOCalculator, equal


class DFTBPlus(FileIOCalculator):
    command = 'dftb+ > PREFIX.out'

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'free_energy', 'charges']

    # NOTE: This is different from default_parameters because applying
    # these default parameters requires extra logic for merging nested dicts.
    _default_params = dict(
        hamiltonian=dict(dftb=dict(
            slaterkosterfiles=dict(type2filenames=dict(
                separator='-',
                suffix='.skf',
            )),
            maxangularmomentum=dict(),
        )),
        parseroptions=dict(
            parserversion=7,
        ),
        options=dict(
            writedetailedout=True,
            writeresultstag=True,
        ),
    )

    def __init__(self, label='dftbplus', **params):
        FileIOCalculator.__init__(self, label=label, **params)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(os.path.join(self.directory, 'dftb_in.hsd'), atoms,
              properties=properties, default_params=self._default_params,
              **self.parameters)

    def read_results(self):
        self.calc = get_dftb_results(self.atoms, self.directory, self.label)
        self.results = self.calc.results

    def merge(self, old=None, **params):
        changed_parameters = {}

        # behaves similar to Calculator.set(), but merges dicts where possible
        if old is None:
            old = self.parameters

        for key, val in enumerate(params):
            if isinstance(old.get(key), dict) and isinstance(val, dict):
                changed_parameters[key] = self.merge(old=old[key], **val)
            elif key not in old or not equal(val, old[key]):
                changed_parameters[key] = val
                old[key] = val

        return changed_parameters

    def band_structure(self):
        return self.calc.band_structure()
