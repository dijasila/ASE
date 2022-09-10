import pytest
import numpy as np
from glob import glob
from ase import io
from ase.optimize.precon import PreconLBFGS
from ase.build import bulk
from ase.units import GPa

calc = pytest.mark.calculator


@calc('abacus')
def test_abacus_C_volrelax(factory):
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
        C = bulk('C')
        calc = factory.calc(calculation='cell-relax', force_thr_ev=0.01,
                            stress_thr=10, cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        C.calc = calc
        C.get_potential_energy()  # Execute

        # Explicitly parse atomic position output file from Abacus
        out_files = glob('OUT.ABACUS/STRU_ION*_D')
        STRU_C = io.read(out_files[-1], format='abacus')

        print('C cell post relaxation from calc:\n',
              calc.get_atoms().get_cell())
        print('C cell post relaxation from atoms:\n', C.get_cell())
        print('C cell post relaxation from OUT.ABACUS:\n', STRU_C.get_cell())

        # Cl the cells should be the same.
        assert (calc.get_atoms().get_cell() == STRU_C.get_cell()).all()
        assert (C.get_cell() == STRU_C.get_cell()).all()

        return C

    # -- Volume relaxation using ASE with Abacus as force/stress calculator
    def ase_vol_relax():
        C = bulk('C', 'fcc', a=4.5, cubic=True)
        calc = factory.calc(calculation='scf', cal_force=1, cal_stress=1, out_stru=1,
                            kpts=kpts, **input_param)
        C.calc = calc

        with PreconLBFGS(C, logfile='relaxation.log') as qn:
            qn.run(fmax=0.1, smax=GPa)

        print('Stress:\n', calc.read_stress())
        print('C post ASE volume relaxation\n', calc.get_atoms().get_cell())

        return C

    # Test function for comparing two cells
    def cells_almost_equal(cellA, cellB, tol=0.01):
        return (np.abs(cellA - cellB) < tol).all()

    C_abacus = abacus_vol_relax()
    C_ase = ase_vol_relax()

    assert cells_almost_equal(C_ase.get_cell(), C_abacus.get_cell())

    # Cleanup
    C_ase.calc.clean()
