""" This module defines an ASE interface to ABACUS.
Created on Fri Jun  8 16:33:38 2018

Modified on Wed Jun 20 15:00:00 2018
@author: Shen Zhen-Xiong

Modified on Wed Jun 03 23:00:00 2022
@author: Ji Yu-yang
"""

import os
import numpy as np

from ase.io import write
from ase.calculators.abacus.create_input import AbacusInput
from ase.calculators.genericfileio import (GenericFileIOCalculator,
                                           CalculatorTemplate)


def get_abacus_version(string):
    import re
    match = re.search(r'Version:\s*(.*)\n', string, re.M)
    return match.group(1)


class AbacusProfile:
    def __init__(self, argv):
        self.argv = argv

    def run(self, directory, outputname):
        # import subprocess
        from subprocess import check_call

        # with open(directory / outputname, "w") as fd:
        #     proc = subprocess.Popen(
        #         self.argv, stdout=fd, stderr=subprocess.PIPE, shell=True, cwd=directory, env=os.environ)

        #     out, err = proc.communicate()

        with open(directory / outputname, "w") as fd:
            check_call(self.argv, stdout=fd, cwd=directory,
                       env=os.environ)


class AbacusTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            "abacus",
            [
                "energy",
                "forces",
                "stress",
                "free_energy",
            ],
        )

        self.outputname = "abacus.out"

    def update_parameters(self, atoms, parameters, properties):
        """Check and update the parameters to match the desired calculation

        Parameters
        ----------
        atoms : atoms.Atoms
            The atoms object to perform the calculation on.
        parameters: dict
            The parameters used to perform the calculation.
        properties: list of str
            The list of properties to calculate

        Returns
        -------
        dict
            The updated parameters object
        """
        parameters = dict(parameters)
        property_flags = {
            "forces": "cal_force",
            "stress": "cal_stress",
        }
        # Ensure ABACUS will calculate all desired properties
        for property in properties:
            abacus_name = property_flags.get(property, None)
            if abacus_name is not None:
                parameters[abacus_name] = 1

        ntype = parameters.get('ntype', None)
        if not ntype:
            numbers = np.unique(atoms.get_atomic_numbers())
            parameters["ntype"] = len(numbers)

        return parameters

    def write_input(self, directory, atoms, parameters, properties):
        """Write the input files for the calculation

        Parameters
        ----------
        directory : Path
            The working directory to store the input files.
        atoms : atoms.Atoms
            The atoms object to perform the calculation on.
        parameters: dict
            The parameters used to perform the calculation.
        properties: list of str
            The list of properties to calculate
        """
        parameters = self.update_parameters(atoms, parameters, properties)

        pseudo_dir = parameters.pop('pseudo_dir', None)
        basis_dir = parameters.pop('orbital_dir') if parameters.get(
            'orbital_dir', None) else parameters.pop('basis_dir', None)

        self.out_suffix = parameters.get(
            'suffix') if parameters.get('suffix', None) else 'ABACUS'
        self.cal_name = parameters.get(
            'calculation') if parameters.get('calculation', None) else 'scf'

        abacus_input = AbacusInput()
        abacus_input.set(**parameters)
        abacus_input.write_input_core(directory=directory)
        abacus_input.write_kpt(directory=directory)
        abacus_input.write_pp(
            pp=parameters['pp'], directory=directory, pseudo_dir=pseudo_dir)
        if 'basis' in parameters.keys():
            abacus_input.write_orb(
                basis=parameters['basis'], directory=directory, basis_dir=basis_dir)
        if 'offsite_basis' in parameters.keys():
            abacus_input.write_abfs(offsite_basis=parameters['offsite_basis'], directory=directory, offsite_basis_dir=parameters.get(
                'offsite_basis_dir', None))

        write(os.path.join(directory, 'STRU'), atoms, format='abacus', pp=parameters['pp'], basis=parameters.get('basis', None),
              offsite_basis=parameters.get('offsite_basis', None), scaled=parameters.get("scaled", True), init_vel=parameters.get("init_vel", True))

    def execute(self, directory, profile):
        profile.run(directory, self.outputname)

    def read_results(self, directory):
        from ase.io import read
        path = directory / ('OUT.' + self.out_suffix)
        atoms = read(path / f'running_{self.cal_name}.log', format='abacus-out')
        return dict(atoms.calc.properties())


class Abacus(GenericFileIOCalculator):
    def __init__(self, profile=None, directory='.', **kwargs):
        """Construct the ABACUS calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' or any of ABACUS'
        native keywords.


        Arguments:

        pp: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.UPF', 'H': 'H.UPF'}``.
            A dummy name will be used if none are given.

        basis: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.orb', 'H': 'H.orb'}``.
            A dummy name will be used if none are given.

        kwargs : dict
            Any of the base class arguments.

        """

        if profile is None:
            profile = AbacusProfile(["abacus"])

        super().__init__(template=AbacusTemplate(),
                         profile=profile,
                         parameters=kwargs,
                         directory=directory)
