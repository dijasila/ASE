"""This module defines an ASE interface to NWchem

https://nwchemgit.github.io
"""
import os

import numpy as np

from ase import io
from ase.calculators.calculator import FileIOCalculator
from ase.spectrum.band_structure import BandStructure
from ase.units import Hartree


class NWChem(FileIOCalculator):
    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
        'dipole',
    ]
    command = 'nwchem PREFIX.nwi > PREFIX.nwo'
    accepts_bandpath_keyword = True
    discard_results_on_any_change = True

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=FileIOCalculator._deprecated,
        label='nwchem',
        atoms=None,
        command=None,
        **kwargs,
    ):
        """
        NWChem keywords are specified using (potentially nested)
        dictionaries. Consider the following input file block::

            dft
                odft
                mult 2
                convergence energy 1e-9 density 1e-7 gradient 5e-6
            end

        This can be generated by the NWChem calculator by using the
        following settings:
        >>> from ase.calculators.nwchem import NWChem
        >>> calc = NWChem(dft={'odft': None,
        ...                    'mult': 2,
        ...                    'convergence': {'energy': 1e-9,
        ...                                    'density': 1e-7,
        ...                                    'gradient': 5e-6,
        ...                                    },
        ...                    },
        ...               )

        In addition, the calculator supports several special keywords:

        theory: str
            Which NWChem module should be used to calculate the
            energies and forces. Supported values are ``'dft'``,
            ``'scf'``, ``'mp2'``, ``'ccsd'``, ``'tce'``, ``'tddft'``,
            ``'pspw'``, ``'band'``, and ``'paw'``. If not provided, the
            calculator will attempt to guess which theory to use based
            on the keywords provided by the user.
        xc: str
            The exchange-correlation functional to use. Only relevant
            for DFT calculations.
        task: str
            What type of calculation is to be performed, e.g.
            ``'energy'``, ``'gradient'``, ``'optimize'``, etc. When
            using ``'SocketIOCalculator'``, ``task`` should be set
            to ``'optimize'``. In most other circumstances, ``task``
            should not be set manually.
        basis: str or dict
            Which basis set to use for gaussian-type orbital
            calculations. Set to a string to use the same basis for all
            elements. To use a different basis for different elements,
            provide a dict of the form:

            >>> calc = NWChem(...,
            ...               basis={'O': '3-21G',
            ...                      'Si': '6-31g'})

        basispar: str
            Additional keywords to put in the NWChem ``basis`` block,
            e.g. ``'rel'`` for relativistic bases.
        symmetry: int or str
            The point group (for gaussian-type orbital calculations) or
            space group (for plane-wave calculations) of the system.
            Supports both group names (e.g. ``'c2v'``, ``'Fm3m'``) and
            numbers (e.g. ``225``).
        autosym: bool
            Whether NWChem should automatically determine the symmetry
            of the structure (defaults to ``False``).
        center: bool
            Whether NWChem should automatically center the structure
            (defaults to ``False``). Enable at your own risk.
        autoz: bool
            Whether NWChem should automatically construct a Z-matrix
            for your molecular system (defaults to ``False``).
        geompar: str
            Additional keywords to put in the NWChem `geometry` block,
            e.g. ``'nucleus finite'`` for gaussian-shaped nuclear
            charges. Do not set ``'autosym'``, ``'center'``, or
            ``'autoz'`` in this way; instead, use the appropriate
            keyword described above for these settings.
        set: dict
            Used to manually create or modify entries in the NWChem
            rtdb. For example, the following settings enable
            pseudopotential filtering for plane-wave calculations::

                set nwpw:kbpp_ray .true.
                set nwpw:kbpp_filter .true.

            These settings are generated by the NWChem calculator by
            passing the arguments:

            >>> calc = NWChem(...,
            >>>               set={'nwpw:kbpp_ray': True,
            >>>                    'nwpw:kbpp_filter': True})

        kpts: (int, int, int), or dict
            Indicates which k-point mesh to use. Supported syntax is
            similar to that of GPAW. Implies ``theory='band'``.
        bandpath: BandPath object
            The band path to use for a band structure calculation.
            Implies ``theory='band'``.
        pretasks: list of dict
            Tasks used to produce a better initial guess
            for the wavefunction.
            These task typically use a cheaper level of theory
            or smaller basis set (but not both).
            The output energy and forces should remain unchanged
            regardless of the number of tasks or their parameters,
            but the runtime may be significantly improved.

            For example, a MP2 calculation preceded by guesses at the
            DFT and HF levels would be

            >>> calc = NWChem(theory='mp2', basis='aug-cc-pvdz',
            >>>               pretasks=[
            >>>                   {'dft': {'xc': 'hfexch'},
            >>>                    'set': {'lindep:n_dep': 0}},
            >>>                   {'theory': 'scf', 'set': {'lindep:n_dep': 0}},
            >>>               ])

            Each dictionary could contain any of the other parameters,
            except those which pertain to global configurations
            (e.g., geometry details, scratch dir).

            The default basis set is that of the final step in the calculation,
            or that of the previous step that which defines a basis set.
            For example, all steps in the example will use aug-cc-pvdz
            because the last step is the only one which defines a basis.

            Steps which change basis set must use the same theory.
            The following specification would perform SCF using the 3-21G
            basis set first, then B3LYP//3-21g, and then B3LYP//6-31G(2df,p)

            >>> calc = NWChem(theory='dft', xc='b3lyp', basis='6-31g(2df,p)',
            >>>               pretasks=[
            >>>                   {'theory': 'scf', 'basis': '3-21g',
            >>>                    'set': {'lindep:n_dep': 0}},
            >>>                   {'dft': {'xc': 'b3lyp'}},
            >>>               ])

            The :code:`'set': {'lindep:n_dep': 0}` option is highly suggested
            as a way to avoid errors relating to symmetry changes between tasks.

            The calculator will configure appropriate options for saving
            and loading intermediate wavefunctions, and
            place an "ignore" task directive between each step so that
            convergence errors in intermediate steps do not halt execution.
        """
        FileIOCalculator.__init__(
            self,
            restart,
            ignore_bad_restart_file,
            label,
            atoms,
            command,
            **kwargs,
        )
        self.calc = None

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Prepare perm and scratch directories
        perm = os.path.abspath(self.parameters.get('perm', self.label))
        scratch = os.path.abspath(self.parameters.get('scratch', self.label))
        os.makedirs(perm, exist_ok=True)
        os.makedirs(scratch, exist_ok=True)

        io.write(
            self.label + '.nwi',
            atoms,
            properties=properties,
            label=self.label,
            **self.parameters,
        )

    def read_results(self):
        output = io.read(self.label + '.nwo')
        self.calc = output.calc
        self.results = output.calc.results

    def band_structure(self):
        self.calculate()
        perm = self.parameters.get('perm', self.label)
        if self.calc.get_spin_polarized():
            alpha = np.loadtxt(os.path.join(perm, self.label + '.alpha_band'))
            beta = np.loadtxt(os.path.join(perm, self.label + '.beta_band'))
            energies = np.array([alpha[:, 1:], beta[:, 1:]]) * Hartree
        else:
            data = np.loadtxt(
                os.path.join(perm, self.label + '.restricted_band')
            )
            energies = data[np.newaxis, :, 1:] * Hartree
        eref = self.calc.get_fermi_level()
        if eref is None:
            eref = 0.0
        return BandStructure(self.parameters.bandpath, energies, eref)
