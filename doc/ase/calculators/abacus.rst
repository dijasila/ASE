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

The ABACUS calculator can be imported through::

  from ase.calculators.abacus import Abacus

The input parameters can be set like::

  calc = Abacus(profile=profile, ntype=1, ecutwfc=50, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', ks_solver='cg', calculation='scf' pp=pp, basis=basis, kpts=kpts)

in which ``pp`` is a dict of pseudopotentials for involved elememts, such as ``pp={'Al':'Al_ONCV_PBE-1.0.upf',...}``; ``basis`` is a dict of orbital files, such as ``basis={'Al':'Al_gga_10au_100Ry_4s4p1d.orb'}``; ``kpts`` is a parameter used to set k-grids.

The command to run jobs can be set by specifying ``AbacusProfile``::

  from ase.calculators.abacus import AbacusProfile
  abacus = '/usr/local/bin/abacus'
  profile = AbacusProfile(argv=['mpirun','-n','2',abacus])

in which ``abacus`` sets the absolute path of the ``abacus`` executable.

For more information on pseudopotentials and numperical orbitals, please visit ABACUS_. The elaboration of input parameters can be found here_.

.. _here: https://github.com/deepmodeling/abacus-develop/blob/develop/docs/input-main.md


