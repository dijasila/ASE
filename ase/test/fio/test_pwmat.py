import io
import os
import unittest

from ase import Atoms
from ase.io import read

# atom.config content
atom_config_string: str = '''  12 atoms
 Lattice vector (Angstrom)
   3.190316E+00      5.525789E+00      0.000000E+00  
   -6.380631E+00     0.000000E+00      0.000000E+00  
   0.000000E+00      0.000000E+00      2.312977E+01  
 Position (normalized), move_x, move_y, move_z
  42         0.333333           0.166667           0.500000       1  1  1
  16         0.166667           0.333333           0.432343       1  1  1
  16         0.166667           0.333333           0.567657       1  1  1
  42         0.333333           0.666667           0.500000       1  1  1
  16         0.166667           0.833333           0.432343       1  1  1
  16         0.166667           0.833333           0.567657       1  1  1
  42         0.833333           0.166667           0.500000       1  1  1
  16         0.666667           0.333333           0.432343       1  1  1
  16         0.666667           0.333333           0.567657       1  1  1
  42         0.833333           0.666667           0.500000       1  1  1
  16         0.666667           0.833333           0.432343       1  1  1
  16         0.666667           0.833333           0.567657       1  1  1
MAGNETIC
42 0.0
16 0.0
16 0.0
42 0.0
16 0.0
16 0.0
42 0.0
16 0.0
16 0.0
42 0.0
16 0.0
16 0.0
'''

# REPORT CONTENT
report_string: str = '''           1           1
 PRECISION  = AUTO       
 JOB       = SCF
 IN.PSP1   = Mo.SG15.PBE.UPF
 IN.PSP2   = S.SG15.PBE.UPF
 IN.ATOM   = atom.config    
Weighted average num_of_PW for all kpoint=                        32866.440
 ************************************
 E_Hxc(eV)         36168.0440410622     
 E_ion(eV)        -79287.9353888918     
 E_Coul(eV)        37531.6123362230     
 E_Hxc+E_ion(eV)  -43119.8913478296     
 NONSCF     1          AVE_STATE_ERROR= 0.1278E+01
 NONSCF     2          AVE_STATE_ERROR= 0.4100E+00
 NONSCF     3          AVE_STATE_ERROR= 0.5897E-01
 NONSCF     4          AVE_STATE_ERROR= 0.1027E-01
 NONSCF     5          AVE_STATE_ERROR= 0.1746E-02
 NONSCF     6          AVE_STATE_ERROR= 0.3660E-03
 iter=   7   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1580784E+01
 err of ug            = 0.1865E-04
 dv_ave, drho_tot     = 0.0000E+00 0.1115E+00
 E_tot(eV)            = -.97107467802733E+04    -.9711E+04
 -------------------------------------------
 iter=   8   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1668484E+00
 err of ug            = 0.1740E-02
 dv_ave, drho_tot     = 0.2351E-01 0.4224E-01
 E_tot(eV)            = -.97012791705132E+04    0.9468E+01
 -------------------------------------------
 iter=   9   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.1843762E+00
 err of ug            = 0.5185E-03
 dv_ave, drho_tot     = 0.1239E-01 0.2571E-01
 E_tot(eV)            = -.97011992106859E+04    0.7996E-01
 -------------------------------------------
 iter=  10   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.6739299E-01
 err of ug            = 0.1810E-03
 dv_ave, drho_tot     = 0.1330E-01 0.1094E-01
 E_tot(eV)            = -.97011506947260E+04    0.4852E-01
 -------------------------------------------
 iter=  11   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1051469E+00
 err of ug            = 0.1704E-03
 dv_ave, drho_tot     = 0.7131E-02 0.1147E-01
 E_tot(eV)            = -.97011441364653E+04    0.6558E-02
 -------------------------------------------
 iter=  12   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.3913156E+00
 err of ug            = 0.7956E-04
 dv_ave, drho_tot     = 0.7296E-02 0.6186E-02
 E_tot(eV)            = -.97011237684088E+04    0.2037E-01
 -------------------------------------------
 iter=  13   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.2498529E+00
 err of ug            = 0.3156E-04
 dv_ave, drho_tot     = 0.7603E-02 0.5397E-02
 E_tot(eV)            = -.97011183876936E+04    0.5381E-02
 -------------------------------------------
 iter=  14   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.5993476E+00
 err of ug            = 0.3795E-04
 dv_ave, drho_tot     = 0.1014E-01 0.4787E-02
 E_tot(eV)            = -.97011108205844E+04    0.7567E-02
 -------------------------------------------
 iter=  15   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.6440977E+00
 err of ug            = 0.4144E-04
 dv_ave, drho_tot     = 0.6350E-02 0.3427E-02
 E_tot(eV)            = -.97011071628981E+04    0.3658E-02
 -------------------------------------------
 iter=  16   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.7845993E+00
 err of ug            = 0.2698E-04
 dv_ave, drho_tot     = 0.1169E-01 0.2450E-02
 E_tot(eV)            = -.97010947399116E+04    0.1242E-01
 -------------------------------------------
 iter=  17   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.8289866E+00
 err of ug            = 0.1184E-04
 dv_ave, drho_tot     = 0.7705E-02 0.2555E-02
 E_tot(eV)            = -.97010894449507E+04    0.5295E-02
 -------------------------------------------
 iter=  18   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.8828054E+00
 err of ug            = 0.1682E-04
 dv_ave, drho_tot     = 0.1111E-01 0.1701E-02
 E_tot(eV)            = -.97010782531264E+04    0.1119E-01
 -------------------------------------------
 iter=  19   ave_lin=  3.1  iCGmth=   3
 Ef(eV)               = -.9222894E+00
 err of ug            = 0.8819E-05
 dv_ave, drho_tot     = 0.1039E-01 0.1447E-02
 E_tot(eV)            = -.97010749755841E+04    0.3278E-02
 -------------------------------------------
 iter=  20   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.9883360E+00
 err of ug            = 0.8409E-05
 dv_ave, drho_tot     = 0.1329E-01 0.1361E-02
 E_tot(eV)            = -.97010669845761E+04    0.7991E-02
 -------------------------------------------
 iter=  21   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.1034416E+01
 err of ug            = 0.7729E-05
 dv_ave, drho_tot     = 0.1561E-01 0.1309E-02
 E_tot(eV)            = -.97010675551896E+04    -.5706E-03
 -------------------------------------------
 iter=  22   ave_lin=  2.6  iCGmth=   3
 Ef(eV)               = -.1060063E+01
 err of ug            = 0.9139E-05
 dv_ave, drho_tot     = 0.1363E-01 0.1270E-02
 E_tot(eV)            = -.97010647495656E+04    0.2806E-02
 -------------------------------------------
 iter=  23   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.1104588E+01
 err of ug            = 0.9866E-05
 dv_ave, drho_tot     = 0.1660E-01 0.1190E-02
 E_tot(eV)            = -.97010616934368E+04    0.3056E-02
 -------------------------------------------
 iter=  24   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.1177740E+01
 err of ug            = 0.1142E-04
 dv_ave, drho_tot     = 0.1747E-01 0.1088E-02
 E_tot(eV)            = -.97010602913740E+04    0.1402E-02
 -------------------------------------------
 iter=  25   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = -.1202988E+01
 err of ug            = 0.5497E-05
 dv_ave, drho_tot     = 0.1582E-01 0.8864E-03
 E_tot(eV)            = -.97010581022804E+04    0.2189E-02
 -------------------------------------------
 iter=  26   ave_lin=  3.2  iCGmth=   3
 Ef(eV)               = -.1256735E+01
 err of ug            = 0.6583E-05
 dv_ave, drho_tot     = 0.1712E-01 0.7097E-03
 E_tot(eV)            = -.97010581505657E+04    -.4829E-04
 -------------------------------------------
'''


class TestReadPWmat(unittest.TestCase):
    def setUp(self):
        self.atom_config_path: str = "atom.config"
        self.report_path: str = "REPORT"
        with open(self.atom_config_path, 'w') as f:
            f.write(atom_config_string)
        with open(self.report_path, 'w') as f:
            f.write(report_string)
            
    def test_read_pwmat(self):
        atoms: Atoms = read("atom.config")
        self.assertEqual(len(atoms), 12)
        
    def test_read_report(self):
        report = read(self.report_path, index=-1)   # Get last step of scf
        self.assertEqual(report.get_potential_energy(), -.97010581505657E+04)
        
    def tearDown(self):
        if os.path.isfile(self.atom_config_path):
            os.remove(self.atom_config_path)
        if os.path.isfile(self.report_path):
            os.remove(self.report_path)

