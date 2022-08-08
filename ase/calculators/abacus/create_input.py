# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:02:37 2018

@author: shenzx

Modified on Wed Jun 01 15:00:00 2022
@author: Ji Yu-yang
"""

from __future__ import print_function
import os
import shutil
import warnings
from os.path import join, exists
import numpy as np

from ase.calculators import calculator
# copyright © Key Lab of Quantum Information, CAS, China
"""This module defines an ASE interface to ABACUS

Developed on the basis of modules by Zhen-Xiong Shen, modified by Yu-yang Ji.
 The path of the directory containing the
 pseudopotential and basis directories (LDA, PBE, SG15, ORBITAL, ...)
 should be set by the enviromental flag $ABACUS_PP_PATH, $ABACUS_ORBITAL_PATH.

The user should also set the enviroment flag
 $ABACUS_SCRIPT pointing to a python script looking

like::
    import os
    exitcode = os.system('abacus')
http://abacus.ustc.edu.cn/
"""

# Parameters list that can be set in INPUT.  -START-
# 1
system_keys = [
    'suffix',              # the name of main output directory
    'ntype',               # atom species number
    'calculation',         # test; scf; relax; nscf; ienvelope; istate;
    'symmetry',            # turn symmetry on or off
    'symmetry_prec',       # accuracy for symmetry
    # devide all processors into kpar groups, and k points will be distributed among each group.
    'kpar',
    # devide all processors into bndpar groups, and bands (only stochastic orbitals now) will be distributed among each group
    'bndpar',
    'latname',             # the name of lattice name
    'init_wfc',            # start wave functions are from 'atomic' or 'file'
    'init_chg',            # start charge is from 'atomic' or file
    'init_vel',            # read velocity from STRU or not
    'nelec',               # input number of electrons
    'tot_magnetization',   # total magnetization of the system
    'dft_functional',      # exchange correlation functional
    'pseudo_type',         # the type pseudo files
    'pseudo_rcut',         # cut-off radius for radial integration
    # 0: use our own mesh to do radial renormalization; 1: use mesh as in QE
    'pseudo_mesh',
    # Used only for nscf calculations. If set to 1, then a memory saving technique will be used for many k point calculations.
    'mem_saver',
    # If set to a positive number, then it specifies the number of threads used for carrying out diagonalization.
    'diago_proc'
    # If set to a natural number, a Cardinal B-spline interpolation will be used to calculate Structure Factor.
    'nbspline'
]
# 2
file_keys = [
    'stru_file',           # the filename of file containing atom positions
    'kpoint_file',         # the name of file containing k points
    # when the program needs to read files such as electron density(SPIN1_CHG) as a starting point, this variables tells the location of the files.
    'read_file_dir'
]
# 3
pw_keys = [
    'ecutwfc',             # energy cutoff for wave functions
    'nx',                  # number of points along x axis for FFT grid
    'ny',                  # number of points along y axis for FFT grid
    'nz',                  # number of points along z axis for FFT grid
    'pw_seed',             # It is the random seed to initialize wave functions
    'pw_diag_thr',         # threshold for eigenvalues is cg/david electron iterations
    # It indicates the maximal iteration number for cg/david method.
    'pw_diag_nmax',
    # It indicates the maximal dimension for the Davidson method.
    'pw_diag_ndim'
]
# 4
lcao_keys = [
    'nb2d',                # 2d distribution of atoms
    # If not equals to 2, then the maximum l channels on LCAO is set to lmaxmax. If 2, then the number of l channels will be read from the LCAO data sets.
    'lmaxmax',
    'lcao_ecut',           # energy cutoff for LCAO
    'lcao_dk',             # delta k for 1D integration in LCAO
    'lcao_dr',             # delta r for 1D integration in LCAO
    'lcao_rmax',           # max R for 1D two-center integration table
    'search_radius',       # input search radius (Bohr)
    'search_pbc',          # input periodic boundary condition
    'noncolin',            # using non-collinear-spin
    'lspinorb'             # consider the spin-orbit interaction
]
# 5
elec_keys = [
    'basis_type',          # PW; LCAO in pw; LCAO
    'ks_solver',                # cg; dav; lapack; genelpa; hpseps; scalapack_gvx
    'nbands',              # number of bands
    'nbands_istate',       # number of bands around Fermi level for istate calulation
    'nspin',               # 1: single spin; 2: up and down spin; 4: noncollinear spin
    # Specifies how to calculate the occupations of bands.
    'occupations',
    'smearing_method',     # type of smearing_method: gauss; fd; fixed; mp; mp2; mv
    'smearing_sigma',      # energy range for smearing
    'mixing_type',         # plain; kerker; pulay; pulay-kerker
    'mixing_beta',         # mixing parameter: 0 means no new charge
    'mixing_ndim',         # mixing dimension in pulay
    'mixing_gg0',          # mixing parameter in kerker
    # It is an important parameter only to be used in localized orbitals set.
    'gamma_only',
    'printe',              # Print out energy for each band for every printe steps
    'scf_nmax',            # number of electron iterations
    'scf_thr',             # charge density error
    'chg_extrap',          # atomic; first-order; second-order; dm:coefficients of SIA
    'nbands_sto',          # number of stochastic bands
    'nche_sto',            # number of orders for Chebyshev expansion in stochastic DFT
    # Trial energy to guess the lower bound of eigen energies of the Hamitonian Operator
    'emin_sto',
    # Trial energy to guess the upper bound of eigen energies of the Hamitonian Operator
    'emax_sto',
    'seed_sto',            # The random seed to generate stochastic orbitals
    'kspacing'             # kspacing
]
# 6
relaxation_keys = [
    'relax_nmax',               # number of ion iteration steps
    'cal_force',               # calculate the force or not
    'force_thr',                # force threshold, unit: Ry/Bohr
    'force_thr_ev',             # force threshold, unit: eV/Angstrom
    'relax_bfgs_w1',            # wolfe condition 1 for bfgs
    'relax_bfgs_w2',            # wolfe condition 2 for bfgs
    'relax_bfgs_rmax',          # maximal trust radius, unit: Bohr
    'relax_bfgs_rmin',          # minimal trust radius, unit: Bohr
    'relax_bfgs_init',          # initial trust radius, unit: Bohr
    'cal_stress',               # calculate the stress or not
    'stress_thr',               # stress threshold
    'press1',                   # target pressure, unit: KBar
    'press2',                   # target pressure, unit: KBar
    'press3',                   # target pressure, unit: KBar
    'fixed_axes',               # which axes are fixed
    'relax_method',             # bfgs; sd; cg; cg_bfgs;
    'relax_cg_thr',             # threshold for switching from cg to bfgs, unit: eV/Angstrom
    'cell_factor'               # used in the construction of the pseudopotential tables
]
# 7
output_keys = [
    'out_force',                # output the out_force or not
    'out_mul',                  # mulliken  charge or not
    # it represents the frequency of electronic iters to output charge density and wavefunction
    'out_freq_elec',
    # it represents the frequency of ionic iters to output charge density and wavefunction
    'out_freq_ion',
    'out_chg',                  # >0 output charge density for selected electron steps
    'out_pot',                  # output realspace potential
    'out_dm',                   # >0 output density matrix
    'out_wfc_pw',               # output wave functions
    'out_wfc_r',                # output wave functions in realspace
    'out_wfc_lcao',             # ouput LCAO wave functions
    'out_dos',                  # output energy and dos
    'out_band',                 # output k-index and band structure
    'out_proj_band',            # output projected band structure
    'out_stru',                 # output the structure files after each ion step
    'out_level',                # ie(for electrons); i(for ions);
    # determines whether to write log from all ranks in an MPI run.
    'out_alllog',
    'out_mat_hs',               # output H and S matrix for each k point in k space
    'out_mat_r',                # output r(R) matrix
    'out_mat_hs2',              # output H(R) and S(R) matrix
    # When set to 1, ABACUS will generate a new directory under OUT.suffix path named as element name
    'out_element_info',
    'restart_save',             # print to disk every step for restart
    'restart_load'              # restart from disk
]
# 8
dos_keys = [
    'dos_emin_ev',      # minimal range for dos
    'dos_emax_ev',      # maximal range for dos
    'dos_edelta_ev',    # delta energy for dos
    'dos_sigma',        # gauss b coefficeinet(default=0.07)
    'dos_scale'         # scale dos range by
]
# 9
deepks_keys = [
    'deepks_out_labels',        # >0 compute descriptor for deepks
    'deepks_descriptor_lmax',   # lmax used in generating descriptor
    'deepks_scf',               # >0 add V_delta to Hamiltonian
    'deepks_model',             # file dir of traced pytorch model: 'model.ptg
    'deepks_bandgap',           # >0 for bandgap label
    # if set 1, prints intermediate quantities that shall be used for making unit test
    'deepks_out_unittest',
    'deepks_descriptor_rcut0',  # rcut used in generating descriptor
    'deepks_descriptor_ecut0'   # ecut used in generating descriptor
]
# 10
dipole_keys = [
    # If set to true, a saw-like potential simulating an electric field is added to the bare ionic potential.
    'efield_flag',
    # a dipole correction is also added to the bare ionic potential.
    'dip_cor_flag',
    # The direction of the electric field or dipole correction is parallel to the reciprocal lattice vector
    'efield_dir',
    # Position of the maximum of the saw-like potential along crystal axis efield_dir
    'efield_pos_max',
    'efield_pos_dec',           # Zone in the unit cell where the saw-like potential decreases
    # Amplitude of the electric field, in Hartree a.u.; 1 a.u. = 51.4220632*10^10 V/m.
    'efield_amp'
]
# 11
exx_keys = [
    'exx_hybrid_alpha',         # fraction of Fock exchange in hybrid functionals,
    'exx_hse_omega',            # range-separation parameter in HSE functional
    'exx_pca_threshold',        # reduce ABFs using principal component analysis
    # Smaller components (less than exx_c_threshold) of the C matrix is neglected to accelerate calculation
    'exx_c_threshold',
    # Smaller values of the V matrix can be truncated to accelerate calculation
    'exx_v_threshold',
    # Smaller values of the density matrix can be truncated to accelerate calculation.
    'exx_dm_threshold',
    # using Cauchy-Schwartz inequality to find an upper bound of each integral before carrying out explicit evaluations.
    'exx_schwarz_threshold',
    # using Cauchy-Schwartz inequality to find an upper bound of each integral before carrying out explicit evaluations.
    'exx_cauchy_threshold',
    # It is related to the cutoff of on-site Coulomb potentials, currently not used.
    'exx_ccp_threshold',
    # This parameter determines how many times larger the radial mesh required for calculating Columb potential is to that of atomic orbitals.
    'exx_ccp_rmesh_times',
    'exx_distribute_type',      # governs the mechanism of atom pairs' distribution
    # The radial part of opt-ABFs are generated as linear combinations of spherical Bessel functions.
    'exx_opt_orb_lmax',
    # A plane wave basis is used to optimize the radial ABFs.
    'exx_opt_orb_ecut',
    # determines the threshold when solving for the zeros of spherical Bessel functions.
    'exx_opt_orb_tolerence'
    'exx_separate_loop',        # way to control exx iteration
    'exx_hybrid_step'           # steps of exx iteration
]
# 12
md_keys = [
    'md_type',            # choose ensemble
    'md_nstep',           # md steps
    'md_ensolver',        # choose potential
    'md_restart',         # whether restart
    'md_dt',              # time step
    'md_tfirst',          # temperature first
    'md_tlast',           # temperature last
    'md_dumpfreq',        # The period to dump MD information
    'md_restartfreq',     # The period to output MD restart information
    'md_tfreq',           # oscillation frequency, used to determine qmass of NHC
    'md_mnhc',            # number of Nose-Hoover chains
    'lj_rcut',            # cutoff radius of LJ potential
    'lj_epsilon',         # the value of epsilon for LJ potential
    'lj_sigma',           # the value of sigma for LJ potential
    'msst_direction',     # the direction of shock wave
    'msst_vel',           # the velocity of shock wave
    'msst_vis',           # artificial viscosity
    'msst_tscale',        # reduction in initial temperature
    'msst_qmass',         # mass of thermostat
    # damping parameter (time units) used to add force in Langevin method
    'md_damp'
]
# 13
dftu_keys = [
    'dft_plus_u',         # calculate plus U correction
    'orbital_corr',       # Specify which orbits need plus U correction for each atom
    'hubbard_u',          # Hubbard Coulomb interaction parameter U(ev)
    'hund_j',             # Hund exchange parameter J(ev)
    # whether use the local screen Coulomb potential method to calculate the value of U and J
    'yukawa_potential',
    'omc'                 # whether turn on occupation matrix control method or not
]
# 14
vdw_keys = [
    # the method of calculating vdw (none ; d2 ; d3_0 ; d3_bj
    'vdw_method',
    'vdw_s6',           # scale parameter of d2/d3_0/d3_bj
    'vdw_s8',           # scale parameter of d3_0/d3_bj
    'vdw_a1',           # damping parameter of d3_0/d3_bj
    'vdw_a2',           # damping parameter of d3_bj
    'vdw_d',            # damping parameter of d2
    'vdw_abc',          # third-order term?
    'vdw_C6_file',      # filename of C6
    'vdw_C6_unit',      # unit of C6, Jnm6/mol or eVA6
    'vdw_R0_file',      # filename of R0
    'vdw_R0_unit',      # unit of R0, A or Bohr
    'vdw_model',        # expression model of periodic structure, radius or period
    'vdw_radius',       # radius cutoff for periodic structure
    'vdw_radius_unit',  # unit of radius cutoff for periodic structure
    'vdw_cn_thr',       # radius cutoff for cn
    'vdw_cn_thr_unit',  # unit of cn_thr, Bohr or Angstrom
    'vdw_period'        # periods of periodic structure
]
# 15
berry_keys = [
    'berry_phase',        # calculate berry phase or not
    'gdir',               # calculate the polarization in the direction of the lattice vector
    'towannier90',        # use wannier90 code interface or not
    'nnkpfile',           # the wannier90 code nnkp file name
    'wannier_spin'        # calculate spin in wannier90 code interface
]
# 16
tddft_keys = [
    'tddft',               # calculate tddft or not
    'td_scf_thr',          # threshold for electronic iteration of tddft
    'td_dt',               # time of ion step
    'td_force_dt',         # time of force change
    'td_val_elec_01',      # td_val_elec_01
    'td_val_elec_02',      # td_val_elec_02
    'td_val_elec_03',      # td_val_elec_03
    'td_vext',             # add extern potential or not
    'td_vext_dire',        # extern potential direction
    'td_timescale',        # extern potential td_timescale
    'td_vexttype',         # extern potential type
    'td_vextout',          # output extern potential or not
    'td_dipoleout',        # output dipole or not
    'ocp',                 # change occupation or not
    'ocp_set'              # set occupation
]
# 17
test_keys = [
    'nurse',            # for coders
    't_in_h',           # calculate the kinetic energy or not
    'vl_in_h',          # calculate the local potential or not
    'vnl_in_h',         # calculate the nonlocal potential or not
    'vh_in_h',          # calculate the hartree potential or not
    'vion_in_h',        # calculate the local ionic potential or not
    'test_force',       # test the force
    'test_stress',      # test the force
    'colour'            # for coders, make their live colourful
]
# 18
isol_keys = [
    'imp_sol',             # calculate implicit solvation correction or not
    'eb_k',                # the relative permittivity of the bulk solvent
    'tau',                 # the effective surface tension parameter
    'sigma_k',             # the width of the diffuse cavity
    'nc_k'                 # the cut-off charge density
]

# Parameters list that can be set in INPUT.  -END-


class AbacusInput:

    # environment variable for PP paths
    ABACUS_PP_PATH = 'ABACUS_PP_PATH'
    if ABACUS_PP_PATH in os.environ:
        pppaths = os.environ[ABACUS_PP_PATH]
    else:
        pppaths = './'

    # environment variable for ORBITAL paths
    ABACUS_ORBITAL_PATH = 'ABACUS_ORBITAL_PATH'
    if ABACUS_ORBITAL_PATH in os.environ:
        orbpaths = os.environ[ABACUS_ORBITAL_PATH]
    else:
        orbpaths = './'

    abfspaths = './'

    # Initialize internal dictionary of input parameters to None  -START-
    def __init__(self, restart=None):
        """Construct the ABACUS calculator."""
        self.system_params = {}
        self.file_params = {}
        self.pw_params = {}
        self.lcao_params = {}
        self.elec_params = {}
        self.relaxation_params = {}
        self.output_params = {}
        self.dos_params = {}
        self.deepks_params = {}
        self.dipole_params = {}
        self.exx_params = {}
        self.dftu_params = {}
        self.vdw_params = {}
        self.md_params = {}
        self.berry_params = {}
        self.tddft_params = {}
        self.test_params = {}
        self.isol_params = {}

        for key in system_keys:
            self.system_params[key] = None
        for key in file_keys:
            self.file_params[key] = None
        for key in pw_keys:
            self.pw_params[key] = None
        for key in lcao_keys:
            self.lcao_params[key] = None
        for key in elec_keys:
            self.elec_params[key] = None
        for key in relaxation_keys:
            self.relaxation_params[key] = None
        for key in output_keys:
            self.output_params[key] = None
        for key in dos_keys:
            self.dos_params[key] = None
        for key in deepks_keys:
            self.deepks_params[key] = None
        for key in dipole_keys:
            self.dipole_params[key] = None
        for key in exx_keys:
            self.exx_params[key] = None
        for key in dftu_keys:
            self.dftu_params[key] = None
        for key in vdw_keys:
            self.vdw_params[key] = None
        for key in md_keys:
            self.md_params[key] = None
        for key in berry_keys:
            self.berry_params[key] = None
        for key in tddft_keys:
            self.tddft_params[key] = None
        for key in test_keys:
            self.test_params[key] = None
        for key in isol_keys:
            self.isol_params[key] = None
        # Initialize internal dictionary of input parameters to None  -END-

        # Appoint the KPT parameters which are not INPUT parameters  -START-
        self.kpt_params = {
            'knumber': 0,           # The number of K points
            'kmode': 'Gamma',       # Mode of K points, can be Gamma, MP, Line, Direct, Cartesian
            'kpts': [1, 1, 1],       # Give the K points
            # # Give the displacement of K points
            'koffset': [0, 0, 0]
        }
        # Appoint the KPT parameters which are not INPUT parameters  -END-

    # Set the INPUT and KPT parameters  -START-
    def set(self, **kwargs):
        for key in kwargs:
            if key in ['xc']:
                self.system_params['dft_functional'] = kwargs['xc']
            elif key in self.system_params:
                self.system_params[key] = kwargs[key]
            elif key in self.file_params:
                self.file_params[key] = kwargs[key]
            elif key in self.pw_params:
                self.pw_params[key] = kwargs[key]
            elif key in self.lcao_params:
                self.lcao_params[key] = kwargs[key]
            elif key in self.elec_params:
                self.elec_params[key] = kwargs[key]
            elif key in self.relaxation_params:
                self.relaxation_params[key] = kwargs[key]
            elif key in self.output_params:
                self.output_params[key] = kwargs[key]
            elif key in self.dos_params:
                self.dos_params[key] = kwargs[key]
            elif key in self.deepks_params:
                self.deepks_params[key] = kwargs[key]
            elif key in self.dipole_params:
                self.dipole_params[key] = kwargs[key]
            elif key in self.exx_params:
                self.exx_params[key] = kwargs[key]
            elif key in self.dftu_params:
                self.dftu_params[key] = kwargs[key]
            elif key in self.vdw_params:
                self.vdw_params[key] = kwargs[key]
            elif key in self.md_params:
                self.md_params[key] = kwargs[key]
            elif key in self.berry_params:
                self.berry_params[key] = kwargs[key]
            elif key in self.tddft_params:
                self.tddft_params[key] = kwargs[key]
            elif key in self.test_params:
                self.test_params[key] = kwargs[key]
            elif key in self.isol_params:
                self.isol_params[key] = kwargs[key]
            elif key in self.kpt_params:
                self.kpt_params[key] = kwargs[key]
            elif key in ['pp', 'basis', 'pseudo_dir', 'basis_dir', 'offsite_basis_dir']:
                continue
            else:
                raise TypeError('Parameter not defined:  ' + key)
    # Set the INPUT and KPT parameters  -END-

    # Write INPUT file  -START-
    def write_input_core(self, directory='./'):
        # TODO: process some parameters separated by ' ' (e.g. ocp_set, hubbard_u ...)
        with open(join(directory, 'INPUT'), 'w') as input_file:
            input_file.write('INPUT_PARAMETERS\n')
            input_file.write('# Created by Atomic Simulation Enviroment\n')
            for key, val in self.system_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.file_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.pw_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.lcao_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.elec_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.relaxation_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.output_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.dos_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.deepks_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.dipole_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.exx_params.items():
                if val is not None:
                    params = str(key) + ' ' * (30 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.dftu_params.items():
                if val is not None:
                    params = str(key) + ' ' * (30 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.vdw_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.md_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.berry_params.items():
                if val is not None:
                    params = str(key) + ' ' * (30 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.tddft_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.test_params.items():
                if val is not None:
                    params = str(key) + ' ' * (20 - len(key)) + str(val)
                    input_file.write('%s\n' % params)

            for key, val in self.isol_params.items():
                if val is not None:
                    params = str(key) + ' ' * (30 - len(key)) + str(val)
                    input_file.write('%s\n' % params)
    # Write INPUT file  -END-
    # Read  INPUT  file  --START-

    def read_input(self, filename='INPUT', **kwargs):
        # TODO: process some parameters separated by ' ' (e.g. ocp_set, hubbard_u ...)
        with open(join(filename), 'r') as file:
            file.readline()
            lines = file.readlines()

        for line in lines:
            try:
                line = line.replace("# ", "#  ")
                data = line.split()
                if len(data) == 0:
                    continue
                elif data[0][0] == "# ":
                    continue

                key = data[0]
                if key in system_keys:
                    self.system_params[key] = data[1]
                elif key in file_keys:
                    self.file_params[key] = data[1]
                elif key in pw_keys:
                    self.pw_params[key] = data[1]
                elif key in lcao_keys:
                    self.lcao_params[key] = data[1]
                elif key in elec_keys:
                    self.lcao_params[key] = data[1]
                elif key in relaxation_keys:
                    self.relaxation_params[key] = data[1]
                elif key in output_keys:
                    self.output_params[key] = data[1]
                elif key in dos_keys:
                    self.dos_params[key] = data[1]
                elif key in deepks_keys:
                    self.deepks_params[key] = data[1]
                elif key in dipole_keys:
                    self.dipole_params[key] = data[1]
                elif key in exx_keys:
                    self.exx_params[key] = data[1]
                elif key in dftu_keys:
                    self.dftu_params[key] = data[1]
                elif key in vdw_keys:
                    if key == 'vdw_period':
                        self.vdw_params[key] = (data[1] + '  '
                                                + data[2] + '  ' + data[3])
                    else:
                        self.vdw_params[key] = data[1]
                elif key in md_keys:
                    self.md_params[key] = data[1]
                elif key in berry_keys:
                    self.berry_params[key] = data[1]
                elif key in tddft_keys:
                    self.tddft_params[key] = data[1]
                elif key in test_keys:
                    self.test_params[key] = data[1]
                elif key in isol_keys:
                    self.isol_params[key] = data[1]

                return 'ok'

            except KeyError:
                raise IOError('keyword "%s" in INPUT is'
                              'not know by calculator.' % key)

            except IndexError:
                raise IOError('Value missing for keyword "%s" .' % key)
    # Read  INPUT  file  --END-

    # Write KPT  -START-
    def write_kpt(self,
                  directory='./',
                  filename='KPT',
                  ):
        k = self.kpt_params

        gamma_only = self.elec_params.get('gamma_only', 0)
        kspacing = self.elec_params.get('kspacing', 0.0) 
        if  gamma_only is not None and gamma_only == 1:
            return 
        if  kspacing is not None and kspacing > 0.0:
            return

        else:
            with open(join(directory, filename), 'w') as kpoint:
                kpoint.write('K_POINTS\n')
                kpoint.write('%s\n' % str(k['knumber']))
                kpoint.write('%s\n' % str(k['kmode']))
                if k['kmode'] in ['Gamma', 'MP']:
                    kpoint.write(' '.join(map(str, k['kpts']))+' '+' '.join(map(str, k['koffset'])))

                elif k['kmode'] in ['Direct', 'Cartesian', 'Line']:
                    for n in range(len(k['kpts'])):
                        for i in range(len(k['kpts'][n])):
                            kpoint.write('%s  ' % str(k['kpts'][n][i]))
                        kpoint.write('\n')

                else:
                    raise ValueError("The value of kmode is not right, set to "
                                     "Gamma, MP, Direct, Cartesian, or Line.")
    # Write KPT  -END-

    # Read KPT file  -START-
    def read_kpt(self,
                 filename='KPT',
                 **kwargs):
        with open(filename, 'r') as file:
            lines = file.readlines()

        if lines[2][-1] == '\n':
            kmode = lines[2][:-1]
        else:
            kmode = lines[2]

        if kmode in ['Gamma', 'MP']:
            self.kpt_params['kmode'] = lines[2][:-1]
            self.kpt_params['knumber'] = lines[1].split()[0]
            self.kpt_params['kpts'] = np.array(lines[3].split()[:3])
            self.kpt_params['koffset'] = np.array(lines[3].split()[3:])

        elif kmode in ['Cartesian', 'Direct', 'Line']:
            self.kpt_params['kmode'] = lines[2][:-1]
            self.kpt_params['knumber'] = lines[1].split()[0]
            self.kpt_params['kpts'] = np.array([list(map(float, line.split()[:3]))
                                                for line in lines[3:]])

        else:
            raise ValueError("The value of kmode is not right, set to "
                             "Gamma, MP, Direct, Cartesian, or Line.")
    # Read KPT file  -END-

    # Copy PP file -START-
    def write_pp(self, pp=None, directory='./', pseudo_dir=None):
        if pseudo_dir:
            self.pppaths = pseudo_dir
        for val in pp.values():
            file = join(self.pppaths, val)
            if not exists(join(directory, val)) and exists(file):
                shutil.copy(file, directory)
            elif exists(join(directory, val)):
                continue
            else:
                raise calculator.InputError(
                    "Can't find pseudopotentials for ABACUS calculation")
    # Copy PP file -END-

    # Copy ORBITAL file -START-
    def write_orb(self, basis=None, directory='./', basis_dir=None):
        if basis_dir:
            self.orbpaths = basis_dir
        for val in basis.values():
            file = join(self.orbpaths, val)
            if not exists(join(directory, val)) and exists(file):
                shutil.copy(file, directory)
            elif exists(join(directory, val)):
                continue
            else:
                raise calculator.InputError(
                    "Can't find basis for ABACUS-lcao calculation")
    # Copy ORBITAL file -END-

    # Copy ABFs file -START-
    def write_abfs(self, offsite_basis=None, directory='./', offsite_basis_dir=None):
        if offsite_basis_dir:
            self.abfspaths = offsite_basis_dir
        for val in offsite_basis.values():
            file = join(self.abfspaths, val)
            if not exists(join(directory, val)) and exists(file):
                shutil.copy(file, directory)
            elif exists(join(directory, val)):
                continue
            else:
                raise calculator.InputError(
                    "Can't find Offsite-ABFs for ABACUS-exx calculation")
    # Copy ORBITAL file -END-
