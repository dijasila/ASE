import pytest
import numpy as np
from glob import glob
from ase import io
from ase.optimize.precon import PreconLBFGS
from ase.build import bulk
from ase.units import GPa

calc = pytest.mark.calculator


@calc('abacus')
def test_abacus_Al_volrelax(factory):
    """
    Run ABACUS tests to ensure that relaxation with the ABACUS calculator works.

    """

    kpts = [4, 4, 4]
    input_param = {'ntype': 1, 'ecutwfc': 20, 'scf_nmax': 50, 'smearing_method': 'gaussian',
                   'smearing_sigma': 0.01, 'basis_type': 'lcao', 'ks_solver': 'genelpa',
                   'mixing_type': 'pulay', 'mixing_beta': 0.7, 'scf_thr': 1e-6, 'out_chg': 1,
                   'xc': 'pbe'}

    # -- Perform Volume relaxation within Vasp
    def abacus_vol_relax():
        Al = bulk('Al', 'fcc', a=4.5, cubic=True)
        calc = factory.calc(calculation='cell-relax', force_thr_ev=0.01,
                            stress_thr=10, cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        Al.calc = calc
        Al.get_potential_energy()  # Execute

        # Explicitly parse atomic position output file from Abacus
        out_files = glob('OUT.ABACUS/STRU_ION*_D')
        STRU_Al = io.read(out_files[-1], format='abacus')

        r_Al = io.read('OUT.ABACUS/running_cell-relax.log',
                       format='abacus-out')

        assert cells_almost_equal(r_Al.get_cell(), STRU_Al.get_cell())

        return r_Al

    # -- Volume relaxation using ASE with Abacus as force/stress calculator
    def ase_vol_relax():
        Al = bulk('Al', 'fcc', a=4.5, cubic=True)
        calc = factory.calc(calculation='scf', cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        Al.calc = calc

        with PreconLBFGS(Al, logfile='relaxation.log', variable_cell=True, precon="Exp") as qn:
            qn.run(fmax=0.1, smax=0.05)

        return Al

    # Test function for comparing two cells
    def cells_almost_equal(cellA, cellB, tol=0.001):
        return (np.abs(cellA - cellB) < tol).all()

    Al_abacus = abacus_vol_relax()
    Al_ase = ase_vol_relax()

    assert cells_almost_equal(Al_ase.get_cell(), Al_abacus.get_cell())

    # Cleanup
    Al_ase.calc.clean()
