from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:33:38 2018

Modified on Wed Jun 20 15:00:00 2018
@author: Shen Zhen-Xiong

Modified on Wed Jun 03 23:00:00 2022
@author: Ji Yu-yang
"""

import os
import subprocess
import numpy as np

from ase.io import write, read
from ase.calculators.abacus.create_input import AbacusInput
from ase.calculators.calculator import FileIOCalculator, PropertyNotPresent

error_template = 'Property "%s" not available. Please try running ABACUS\n' \
                 'first by calling Atoms.get_potential_energy().'


class Abacus(AbacusInput, FileIOCalculator):
    # Initialize parameters and get some information -START-
    name = 'abacus'

    implemented_properties = [
        'energy', 'free_energy', 'forces', 'fermi', 'stress',
        'magmom', 'magmoms'
    ]

    default_parameters = dict(calculation='scf',
                              ecutwfc=50,
                              smearing_method='gaussian',
                              mixing_type='pulay-kerker',
                              basis_type='lcao',
                              gamma_only=1,
                              ks_solver="genelpa",
                              stru_file='STRU',
                              )

    def __init__(self,
                 restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 directory='.',
                 label='abacus',
                 atoms=None,
                 command=None,
                 txt='abacus.out',
                 **kwargs):

        self.results = {}

        # Initialize parameter dictionaries
        AbacusInput.__init__(self, restart)

        # Set directory and label
        self.directory = directory
        self.label = label

        FileIOCalculator.__init__(self,
                                  restart,
                                  ignore_bad_restart_file,
                                  label,
                                  atoms,
                                  **kwargs)

        self.command = command
        self.txt = txt

    # Initialize parameters and get some information -END-

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        Abacus Calculator, then call the create_input.set()
        on remaining inputs for ABACSU specific keys.

        Allows for setting ``label``, ``directory`` and ``txt``
        without resetting the results in the calculator.
        """
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.get('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects
            self.directory = str(kwargs.get('directory'))

        if 'txt' in kwargs:
            self.txt = kwargs.get('txt')

        if 'atoms' in kwargs:
            atoms = kwargs.get('atoms')
            self.atoms = atoms  # Resets results

        if 'command' in kwargs:
            self.command = kwargs.get('command')

        changed_parameters.update(FileIOCalculator.set(self, **kwargs))

        # We might at some point add more to changed parameters, or use it
        if changed_parameters:
            self.reset()  # We don't want to clear atoms
        if kwargs:
            # If we make any changes to Abacus input, we always reset
            AbacusInput.set(self, **kwargs)
            self.results.clear()

    def set_atoms(self, atoms):
        self.atoms = atoms

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore boundary conditions:
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        return system_changes

    def initialize(self, atoms):
        numbers = np.unique(atoms.get_atomic_numbers())
        self.system_params["ntype"] = len(numbers)

    def write_input(self, atoms, properties=None, system_changes=None, scaled=None, set_vel=False, set_mag=False):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        if scaled is None:
            scaled = np.all(atoms.get_pbc())

        self.initialize(atoms)
        AbacusInput.write_input_core(self, directory=self.directory)
        AbacusInput.write_kpt(self, directory=self.directory)
        AbacusInput.write_pp(
            self, pp=self.parameters['pp'], directory=self.directory, pseudo_dir=self.parameters.get('pseudo_dir', None))
        if 'basis' in self.parameters.keys():
            AbacusInput.write_orb(
                self, basis=self.parameters['basis'], directory=self.directory, basis_dir=self.parameters.get('basis_dir', None))
        if 'offsite_basis' in self.parameters.keys():
            AbacusInput.write_abfs(self, offsite_basis=self.parameters['offsite_basis'], directory=self.directory, offsite_basis_dir=self.parameters.get(
                'offsite_basis_dir', None))

        write(os.path.join(self.directory, 'STRU'), atoms, format='abacus', pp=self.parameters['pp'], basis=self.parameters.get('basis', None),
              offsite_basis=self.parameters.get('offsite_basis', None), scaled=scaled, set_vel=False, set_mag=False)

    def read_results(self):
        out_dir = 'OUT.ABACUS' if 'suffix' not in self.parameters.keys(
        ) else self.parameters['suffix']
        cal = 'scf' if 'calculation' not in self.parameters.keys(
        ) else self.parameters['calculation']
        output = read(os.path.join(
            out_dir, f'running_{cal}.log'), format='abacus-out')
        self.calc = output.calc
        self.results = output.calc.results

    def get_fermi_level(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Fermi level')
        return self.calc.get_fermi_level()

    def get_ibz_k_points(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'IBZ k-points')
        ibzkpts = self.calc.get_ibz_k_points()
        return ibzkpts

    def get_k_point_weights(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'K-point weights')
        k_point_weights = self.calc.get_k_point_weights()
        return k_point_weights

    def get_eigenvalues(self, **kwargs):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Eigenvalues')
        eigenvalues = self.calc.get_eigenvalues(**kwargs)
        return eigenvalues

    def get_number_of_spins(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Number of spins')
        nspins = self.calc.get_number_of_spins()
        return nspins

    def band_structure(self, efermi=0.0):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure
        return get_band_structure(calc=self, reference=efermi)

    def run(self):
        with open(self.txt, 'a') as f:
            run = subprocess.Popen(self.command,
                                   stderr=f,
                                   stdin=f,
                                   stdout=f,
                                   cwd=self.directory,
                                   shell=True)
            return run.communicate()
