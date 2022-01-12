.. module:: ase.calculators.rescu

========
RESCU+
========

.. image:: ../../static/rescuplus.png

`RESCU+ <https://www.nanoacademic.com/product-page/rescu-1/>`_ is a density-functional
theory code solving the Kohn-Sham equation using norm-conserving pseudopotentials
and numerical atomic orbital.

The ASE `rescuplus` calculator is an interface to the ``rescuplus_scf`` executable.

Setup
=====

Set up the calculator like a standard ``FileIOCalculator``:

 * ``export ASE_RESCUPLUS_COMMAND="mpiexec -n 1 rescuplus_scf -i PREFIX.rsi > resculog.out && cp rescuplus_scf_out.json PREFIX.rso"``

Any calculation will require pseudopotentials for the elements involved. 
A database in the "nano" format may be found here_.

.. _here: https://storage.googleapis.com/rescumat/TM_LDA_Nano.tar.gz

Untar somewhere and set the environment variable RESCUPLUS_PSEUDO to that directory.
Otherwise, the path to the pseudopotentials should be specified in a dictionary as follows::

    pp_dict = [{"label":"Ga", "path":"Ga_AtomicData.mat"}, {"label":"As", "path":"As_AtomicData.mat"}]


A simple calculation can be set up::

    from ase.build import bulk
    from ase.calculators.rescu import Rescuplus
    from ase.optimize import BFGS
    import os
    a = 5.43 # lattice constant in ang
    atoms = bulk("Si", "diamond", a=a)
    atoms.rattle(stdev=0.05, seed=1) # move atoms around a bit
    # rescu calculator
    # Nanobase PP Si_AtomicData.mat should be found on path RESCUPLUS_PSEUDO (env variable)
    inp = { "system": { "cell" : {"resolution": 0.20}, "kpoint" : {"grid":[5,5,5]}}}
    inp["energy"] = {"forces_return": True, "stress_return": True}
    inp["solver"] = {"mix": {"alpha": 0.5}, "restart": {"DMRPath": "nano_scf_out.h5"}}
    cmd = "mpiexec -n 4 rescuplus_scf -i PREFIX.rsi > resculog.out && cp nano_scf_out.json PREFIX.rso"
    atoms.calc = Rescuplus(command=cmd, input_data=inp)
    # relaxation calculation
    os.system("rm -f nano_scf_out.h5")
    opt = BFGS(atoms, trajectory="si2.traj", logfile="relax.out")
    opt.run(fmax=0.01)

Parameters
==========

Any keyword accepted by RESCU+ may be passed in the input_data dictionary.

The units will be automatically converted back and forth between atomic and SI.

Rescuplus Calculator Class
=========================

.. autoclass:: ase.calculators.rescu.Rescuplus

