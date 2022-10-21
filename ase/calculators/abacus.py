""" This module defines an ASE interface to ABACUS.
Created on Fri Jun  8 16:33:38 2018

Modified on Wed Jun 20 15:00:00 2018
@author: Shen Zhen-Xiong

Modified on Wed Jun 03 23:00:00 2022
@author: Ji Yu-yang
"""

import os
import numpy as np

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
        from subprocess import check_call

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
                "magmom"
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

        from ase.io.abacus import write_kpt, copy_pp, copy_basis, copy_offsite_basis, write_input

        write_input(open(directory / 'INPUT', 'w'), parameters=parameters)
        write_kpt(open(directory / 'KPT', 'w'), parameters=parameters)
        copy_pp(parameters['pp'].values(), pseudo_dir, directory)
        if 'basis' in parameters.keys():
            copy_basis(parameters['basis'].values(), basis_dir, directory)
        if 'offsite_basis' in parameters.keys():
            copy_offsite_basis(parameters['offsite_basis'].values(
            ), parameters.get('offsite_basis_dir', None), directory)

        from ase.io.abacus import write_abacus
        write_abacus(open(directory / 'STRU', 'w'), atoms, pp=parameters['pp'], basis=parameters.get('basis', None),
                     offsite_basis=parameters.get('offsite_basis', None), scaled=parameters.get("scaled", True), init_vel=parameters.get("init_vel", True))

    def execute(self, directory, profile):
        profile.run(directory, self.outputname)

    def read_results(self, directory):
        from ase.io.abacus import read_abacus_results

        path = directory / ('OUT.' + self.out_suffix)
        return read_abacus_results(open(path / f'running_{self.cal_name}.log', 'r'), index=-1)[0]


class Abacus(GenericFileIOCalculator):
    def __init__(self, profile=None, directory='.', **kwargs):
        """Construct the ABACUS calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' or any of ABACUS'
        native keywords.


        Parameters
        ----------
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


def spap_analysis(atoms_list, i_mode=4, symprec=0.1, **kwargs):
    """Structure Prototype Analysis Package used here to analyze symmetry and compare similarity of large amount of atomic structures.

    Description:

    The structure prototype analysis package (SPAP) can analyze symmetry and compare similarity of a large number of atomic structures. 
    Typically, SPAP can analyze structures predicted by CALYPSO (www.calypso.cn). We use spglib to analyze symmetry. The coordination 
    characterization function (CCF) is used to measure structural similarity. We developed a unique and advanced clustering method to 
    automatically classify structures into groups. If you use this program and method in your research, please read and cite the publication: 

        Su C, Lv J, Li Q, Wang H, Zhang L, Wang Y, Ma Y. Construction of crystal structure prototype database: methods and applications. 
        J Phys Condens Matter. 2017 Apr 26;29(16):165901. doi: 10.1088/1361-648X/aa63cd 


    Installation:

        1. Use command: `pip install spap`
        2. Download the source code from https://github.com/chuanxun/StructurePrototypeAnalysisPackage, then install with command `python3 setup.py install`

    Parameters
    ----------
    atoms_list: list
        A list of Atoms objects

    i_mode: int
        Different functionality of SPAP.

    symprec: float
        This precision is used to analyze symmetry of atomic structures.

    **kwargs:
        More parameters can be found in ase.calculators.abacus.spap


    .. note::
           SPAP can be used as follows:


           1. Suppose the `directory` contains many ABACUS STRU files,
              you can use `ase.io.read` to get a list of `Atoms` objects:

              >>> path = {< directory >}
              >>> atoms_list = [read(os.path.join(path, file), index=-1, format='abacus') for file in os.listdir(path)]

           2. Perform `spap_analysis` and some files will be output, such as 'Analysis_Output.dat', 'structure_info.csv' and so on:

              >>> spap_analysis(atoms_list)

    """
    try:
        from spap import run_spap
    except ImportError:
        raise ImportError(
            "If you want to use SPAP to analyze symmetry and compare similarity of atomic structures, Please install it first!")

    kwargs.pop('structure_list', None)
    run_spap(symprec=symprec, structure_list=atoms_list,
             i_mode=i_mode, **kwargs)
