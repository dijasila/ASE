.. _metadynamics:

=================================
ASE for  Metadynamics Simulations
=================================

This tutorial shows how to use the :mod:`~ase.calculators.plumed` calculator
for implementing Well-Tempered Metadynamics and computing Collective Variables 
from a trajectory.

Plumed actually allows several actions besides of what we show here. For further 
description of plumed details, visit `this page <http://www.plumed.org/doc>`_. 
The use of other tools of Plumed with ASE is absolutely analogous to what 
it is explained here.

.. contents::

.. _Theory section:

Theory
------

Collective Variables
====================

In the most of the cases, it is impossible to extract clear information about 
the system of interest by monitoring the coordinates of all atoms directly, 
even more if our system contains many atoms. Instead of that, it
is possible to make the monitoring simpler by defining functions of those 
coordinates that describe the chemical properties that we are interested in. 
Those functions are called Collective Variables (CVs) and allows biasing 
specific degrees of freedom or analyzing how those properties evolve. Plumed 
has numerous CVs already implemented that can be used with ASE. For a 
complete explanation of CVs implemented in Plumed, go 
`here <https://www.plumed.org/doc-v2.7/user-doc/html/colvarintro.html>`_.

Metadynamics
============

:doi:`Metadynamics <10.1038/s42254-020-0153-0>` is an enhanced sampling method 
that allows exploring the configuration landscape by adding cumulative bias in 
terms of some CVs. This bias is added each `\tau` time lapse and usually its 
shape is Gaussian. In time t, the accumulated bias is
defined as:


.. math:: 
   \begin{align}
       V_{B}({\bf{s}}, t) = \sum_{t'=\tau, 2\tau,...}^{t'<t}W(t') 
                            \hspace{0.1cm}
                            exp\left({-\sum_i\frac{[s_i\hspace{0.1cm} - 
                            \hspace{0.1cm}
                            s_i(t')]^2}{2\sigma_i}}\right) 
                            \label{bias} \tag{1}
   \end{align}


Where **s** is a set of collective variables, `\sigma_i` is the width of the 
Gaussian related with the i-th collective variable, and *W(t')* is the height 
of the Gaussian in time *t'*. In simple metadynamics, *W(t')* is a constant, 
but in Well-Tempered Metadynamics, the height of the Gaussians is lower where 
previous bias was added. This reduction of height of the new Gaussians decreases 
the error and avoids exploration towards high free energy states that are 
thermodynamically irrelevant. The height in time t' for Well-Tempered 
Metadynamics is defined as:

.. math:: 
   \begin{align}
       W(t') = W exp\left({-\frac{\beta \hspace{0.1cm} V_B({\bf s}, 
                  \hspace{0.1cm}t')}{\gamma}}\right) \label{hills} \tag{2}
   \end{align}

Here, *W* is the maximum height of the Gaussians, `\beta` is the inverse of
the thermal energy (:math:`1/k_BT`) and `\gamma` is a bias factor greater than
one that regulates how fast the height of the bias decreases. The higher the 
bias factor, the slower is the decreasing of the heights. Note that when 
`\gamma` approaches infinity, this equation becomes constant and simple 
metadynamics is recovered. On contrast, when `\gamma` approaches zero, no bias is
added, which is the case of Molecular Dynamics.

In this way, the force with bias acting over the i-th atom becomes in:

.. math::
   \begin{align}
     {\bf F^B}_i = {\bf F}_i - \frac{\partial {\bf s}}{\partial {\bf R}_i} 
                   \frac{\partial V_B({\bf s}, t)}{\partial {\bf s}}
                   \label{bias-force} \tag{3}
   \end{align}

where :math:`{\bf F}_i` is the force over the atom i due to interactions with 
the rest of atoms and the second term is the additional force due to the added 
bias.

Part of the power of metadynamics is that it can be used for exploring 
conformations and the accumulated bias converges to the free energy surface 
(`F({\bf s})`). In the case of Well-Tempered Metadynamics:

.. math::
   \begin{align}
      \lim_{t\rightarrow \infty} V_B ({\bf{s}}, t) = -\frac{(\gamma -1)}
                                                      {\gamma} F({\bf s})
   \end{align}


Planar 7-Atoms Cluster
----------------------

Let's consider a simple system formed by seven atoms with Lennard-Jones (LJ) 
interactions in a planar space. This simple model is presented in the 
`Plumed Masterclass 21.2 <https://www.plumed.org/doc-v2.7/user-doc/html/masterclass-21-2.html#masterclass-21-2-ex-9>`_.
The LJ cluster has several stable isomers (:numref:`fig1`), which can be 
distinguished in a space of the CVs second (SCM) and third (TCM) central 
moments of the distribution of coordinations (red stars in :numref:`fig2`).

.. _fig1:

.. figure:: cluster.png
   :width: 500
   :align: center

   Stable isomers of 7 atoms with Lennard-Jones interactions in a planar 
   space.

The n-th central moment `\mu_n` of an N-atoms cluster is defined as

.. math::
   \begin{equation}
      {\mu_n} = \frac{1}{N} \sum_{i=1}^{N} \left( {X}_{i} - 
                \left< {X} \right> \right)^n
   \end{equation}

where `\left< {X} \right>` is the mean value of `X_i`, which is the 
coordination of the i-th atom:

.. math::
   \begin{equation}
      X_i= \sum_{i\ne j}\frac{1-(r_{ij}/d)^8}{1-(r_{ij}/d)^{16}}
   \end{equation}

For this example, d is fixed to 1.5 `\sigma`, in LJ units.

Molecular Dynamics Simulation
=============================

For showing that it is necessary to use an enhanced sampling method,
let's start with a Langevin simulation without bias. In LJ dimensionless 
reduced units (with `\epsilon` = 1 eV, `\sigma` = 1 :math:`\textrm Å` and 
m = 1 a.m.u), the parameters of the simulation are  `k_\text{B}T=0.1`, 
friction coefficient fixed equal to 1 and a time step of 0.005.

It is supposed that the system should explore all the space of configurations 
due to thermal fluctuations. However, we can see that the system remains in the 
same state, even when we sample for a long time lapse. That is because a 
complete exploration of the configuration 
space could take more time than the possible to simulate. :numref:`fig2` -blue 
dots- shows the trajectory obtained from the following code of unbiased 
Molecular dynamics:

.. literalinclude:: MD.py

This simulation starts from the configuration of minimum energy, whose 
coordinates are imported from :download:`isomer.xyz`, and the 
system remains moving around that state; it does not jump to the other 
isomers. This means we do not obtain a complete sampling of possible 
configurations as mentioned before. Then, an alternative to observe transitions
is to use an enhanced sampling method. In this case, we implement Well-Tempered 
Metadynamics.


.. warning::
   Note that in the plumed set-up, there is a line with the keyword UNITS,
   which is necessary because all parameters in the plumed set-up and output 
   files are assumed to be in plumed internal units. Then, this line is 
   important to mantain the units of all plumed parameters and outputs in ASE
   units. You can ignore this line if you are aware of the units conversion.

Well-Tempered Metadynamics Simulation
=====================================

Well-Tempered Metadynamics method is described in the `Theory section`_. It basically adds external energy 
for pushing the system to explore different conformations. This makes necessary
to add a restrain to avoid that the extra energy dissolves the atoms in vacuum. 
This restriction consists in a semi-harmonic potential with this form:

.. math::
   \begin{equation}
       V(d_i)=\left\{
          \begin{array}{ll}
              100 (d_i - 2)^2 & \text{if }d_i>2 \\
              0               & \text{otherwise}
          \end{array}
          \right.
   \end{equation}

Where `d_i` is the distance of each atom to the center of mass. Note that this 
potential does not do anything whereas the distance between the atom and the 
center of mass is lower than 2 (in LJ dimensionless reduced units), but if it 
is greater (trying to escape), this potential begins to work and send it back 
to be close the other atoms. This is defined with the keyword UPPER_WALLS in 
the plumed set up.

Well-Tempered Metadynamics simulation for this case can be run using this 
script:

.. literalinclude:: MTD.py

Note that Well-Tempered Metadynamics requires the value of the temperature 
according to equation :math:`\ref{hills}`. Then, it is necessary to define the 
kT argument of the calculator in ASE units. SIGMA and PACE are the 
standard deviation of the Gaussians and the deposition interval in terms of 
number of steps (`\tau` in the equation :math:`\ref{bias}`). HEIGHT and 
BIASFACTOR are the maximum height of the Gaussians (W) and the `\gamma` factor 
of the equation :math:`\ref{hills}`, respectively. 

In this case, the Lennard-Jones calculator computes the forces between atoms, 
namely, :math:`{\bf F}_i` forces in equation :math:`\ref{bias-force}`. 
Likewise, you could use your preferred calculator.

.. _fig2:

.. figure:: trajectory.png
   :width: 400
   :align: center

   Trajectories obtained after 10000 steps of unbiased molecular dynamics 
   (blue dots) and metadynamics (orange dots) in the space of second (SCM) and 
   third (TCM) central moments of coordinations. Red stars represent the values
   of these collective variables for the stable isomers shown in :numref:`fig1`.
   
In contrast to the MD case, Metadynamics achieves moving the system towards other states 
in this short simulation. In a longer simulation, it is possible to obtain a 
complete exploration of the different states and to use the accumulated bias 
to reconstruct the free energy. 

When one runs a metadynamics simulation, Plumed generates a file 
called HILLS that contains the information of the deposited Gaussians. You can 
reconstruct the free energy by yourself or can use the plumed tool 
`sum_hills <https://www.plumed.org/doc-v2.7/user-doc/html/sum_hills.html>`_. 
The simplest way of using it is::

    $ plumed sum_hills --hills HILLS

After this, PLUMED creates a fes.dat file with the FES reconstructed.


Restart
=======

Suppose you realized it was not enough added bias when it finalized. Then, you 
have to restart your simulation during some steps more. For doing so, you have 
to configure the atoms object in the last state of the previous simulation and to 
fix the value of steps (isteps) in the plumed calculator. Taking the last code 
as an example, this means you would have to change the definition of the object 
atoms as follows::

    from ase.io import read

    last_configuration = read('MTD.traj')
    atoms.set_positions(last_configuration.get_positions())
    atoms.set_momenta(last_configuration.get_momenta())

and the definition of the calculator becomes in::
    
    atoms.calc = Plumed( ... , restart=True)
    atoms.calc.istep = 10000

Alternatively, you can use the next function:

.. autoclass:: ase.calculators.plumed.restart_from_trajectory

Post Processing
===============

If you have a trajectory, you can reconstruct the plumed files without 
running again all the simulation. As an example, let's use the trajectory
created with the MD code for rewriting the COLVAR file:

.. literalinclude:: postpro.py
