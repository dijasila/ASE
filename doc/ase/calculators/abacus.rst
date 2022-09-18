.. module:: ase.calculators.abacus

======
ABACUS
======

Introduction
============

ABACUS_ (Atomic-orbital Based Ab-initio Computation at UStc) is an open-source computer code package aiming for large-scale electronic-structure simulations from first principles, developed at the Key Laboratory of Quantum Information and Supercomputing Center, University of Science and Technology of China (USTC) - Computer Network and Information Center, Chinese of Academy (CNIC of CAS). 

.. _ABACUS: http://abacus.ustc.edu.cn/

Environment variables
=================

ABACUS_ supports three types of basis sets: pw, LCAO, and LCAO-in-pw. The path of pseudopotential and numerical orbital files can be set throught the environment variables ``ABACUS_PP_PATH`` and ``ABACUS_ORBITAL_PATH``, respectively, e.g.::

  PP=${HOME}/pseudopotentials
  ORB=${HOME}/orbitals
  export ABACUS_PP_PATH=${PP}
  export ABACUS_ORBITAL_PATH=${ORB}
 
For pw calculations, only ``ABACUS_PP_PATH`` is needed. For LCAO and LCAO-in-pw calculations, both ``ABACUS_PP_PATH`` and ``ABACUS_ORBITAL_PATH`` should be set.


ABACUS Calculator
=================

The default initialization command for the ABACUS calculator is

.. autoclass:: Abacus

In order to run a calculation, you have to ensure that at least the following parameters are specified, either in the initialization or as environment variables:

===============  ====================================================
keyword          description
===============  ====================================================
``pp``            dict of pseudopotentials for involved elememts, 
                  such as ``pp={'Al':'Al_ONCV_PBE-1.0.upf',...}``.
``pseudo_dir``    directory where the pseudopotential are located, 
                  Can also be specified with the ``ABACUS_PP_PATH``
                  environment variable. Default: ``pseudo_dir=./``.
``basis``         dict of orbital files for involved elememts, such as 
                  ``basis={'Al':'Al_gga_10au_100Ry_4s4p1d.orb'}``.
                  It must be set if you want to do LCAO and LCAO-in-pw 
                  calculations. But for pw calculations, it can be omitted.
``basis_dir``     directory where the orbital files are located, 
                  Can also be specified with the ``ABACUS_ORBITAL_PATH``
                  environment variable. Default: ``basis_dir=./``.
``xc``            which exchange-correlation functional is used.
                  An alternative way to set this parameter is via
                  seting ``dft_functional`` which is an ABACUS
                  parameter used to specify exchange-correlation 
                  functional
``kpts``          a tuple (or list) of 3 integers ``kpts=(int, int, int)``, 
                  it is interpreted as the dimensions of a Monkhorst-Pack 
                  grid, when ``kmode`` is ``Gamma`` or ``MP``. It is 
                  interpreted as k-points, when ``kmode`` is ``Direct``,
                  ``Cartesian`` or ``Line``, and ``knumber`` should also
                  be set in these modes to denote the number of k-points.
                  Some other parameters for k-grid settings:
                  including ``koffset`` and ``kspacing``.
===============  ====================================================

For more information on pseudopotentials and numerical orbitals, please visit ABACUS_. The elaboration of input parameters can be found here_.

.. _here: https://github.com/deepmodeling/abacus-develop/blob/develop/docs/input-main.md

The input parameters can be set like::

  calc = Abacus(profile=profile, ntype=1, ecutwfc=50, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', ks_solver='cg', calculation='scf' pp=pp, basis=basis, kpts=kpts)

The command to run jobs can be set by specifying ``AbacusProfile``::

  from ase.calculators.abacus import AbacusProfile
  abacus = '/usr/local/bin/abacus'
  profile = AbacusProfile(argv=['mpirun','-n','2',abacus])

in which ``abacus`` sets the absolute path of the ``abacus`` executable.


SPAP Analysis
=================

SPAP_ (Structure Prototype Analysis Package) is written by Dr. Chuanxun Su to analyze symmetry and compare similarity of large amount of atomic structures. The coordination characterization function (CCF) is used to 
measure structural similarity. An unique and advanced clustering method is developed to automatically classify structures into groups. 

.. _SPAP: https://github.com/chuanxun/StructurePrototypeAnalysisPackage

If you use this program and method in your research, please read and cite the publication:

:doi:`Su C, Lv J, Li Q, Wang H, Zhang L, Wang Y, Ma Y. Construction of crystal structure prototype database: methods and applications. J Phys Condens Matter. 2017 Apr 26;29(16):165901 <10.1088/1361-648X/aa63cd>`.

.. autofunction:: spap_analysis
