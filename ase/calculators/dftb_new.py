import os
from ase.io import write
from ase.io.dftb import get_dftb_results
from ase.calculators.calculator import FileIOCalculator


class DFTBPlus(FileIOCalculator):
    command = 'dftb+ > PREFIX.out'

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'free_energy', 'charges']

    # NOTE: This is different from default_parameters because applying
    # these default parameters requires extra logic for merging nested dicts.
    _default_params = dict(
        hamiltonian=('dftb', dict(
            slaterkosterfiles=('type2filenames', dict(
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

    def band_structure(self):
        return self.calc.band_structure()
