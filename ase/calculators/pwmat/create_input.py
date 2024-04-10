"""This module defines an ASE interface to PWmat."""
from __future__ import annotations
import os
import shutil
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from ase.atoms import Atoms
from ase.config import cfg
from ase.calculators.pwmat.etot_writer import write_etot_input
from ase.calculators.calculator import ReadError


__author__ = "Hanyu Liu"
__email__ = "domainofbuaa@gmail.com"
__date__ = "2024-4-7"


FLOAT_FORMAT = '5.6f'
EXP_FORMAT = '5.2e'

### Type of params
parallel_keys: List[str] = ["parallel"]
### Type 1. float
float_keys: List[str] = [
    'Ecut',
    'Ecut2',
    'Ecut2L',
    'EcutP',
    'HSE_OMEGA',
    'HSE_ALPHA',
    'HSE_BETA',
    'LONDON_S6',
    'LONDON_C6(1)',
    'LONDON_C6(2)',
    'LONDON_RCUT',
    'DFTD3_S6',
    'DFTD3_RS6',
    'DFTD3_S18',
    'DFTD3_RS18',
    'DFTD3_ALPHA6',
    'NUM_ELECTRON',
    'WG_ERROR',
    'E_ERROR',
    'RHO_ERROR',
    'RHO_RELATIVE_ERROR',
    'FORCE_RELATIVE_ERROR',
    'RCUT',
    'IN.PSP_RCUT1',
    'IN.PSP_RCUT2',
    'KERK_AMIN',
    'KERK_AMIX',
    'KERK_AMIX_MAG',
    'KERK_BMIX',
    'LDAU_MIX',
    'PULAY_WEIGHT_SPIN',
    'PULAY_WEIGHT_NS',
    'DOS_GAUSSIAN_BROADENING',      # Set when calculating DOS.
]

### Type 2. int
int_keys: List[str] = [
    'SPIN',
    'CONSTRAINT_MAG',
    'DFTD3_VERSION',
    'COULOMB',
    'OUT.REAL.RHOWF_SP',
    'NUM_KPT',
    'NUM_BAND',
    'SYS_TYPE',
    'NONLOCAL',
    'MD_VV_SCALE',
    'NUM_BLOCKED_PSI',
    'WF_STORE2DISK',
    'NUM_DOS_GRID',
    'NMAP_MAX',
    'NUM_MPI_PER_GPU',
    'FLAG_CYLINDER',
    'NUM_DOS_GRID',               # Set when calculating DOS
]

### Type 3. bool ('T' / 'F')
bool_keys: List[str] = [
    'PWSCF_OUTPUT',
    'HSE_SAVE_CPU_MEMORY',
    'DFTD3_3BODY',
    'IN.WG',
    'OUT.WG',
    'IN.RHO',
    'OUT.RHO',
    'IN.VR',
    'OUT.VR',
    'IN.VEXT',
    'OUT.VATOM',
    'OUT.FORCE',            # Calculate the atomic force and output it to OUT.FORCE file.
    'OUT.STRESS',           # Calculate the virial tensor and output it to OUT.STRESS file.
    'IN.SYMM',
    'CHARGE_DECOMP',        # Output the charge of each atom
    'ENERGY_DECOMP',        # Output the energy of each atom
    'IN.SOLVENT',
    'IN.NONSCF',
    'IN.CC',
    'IN.OCC_ADIA',
    'OUT.MLMD',
    'OUT.RHOATOM',
    'IN.KPT',               # Input path for nonscf
]

### Type 4. char
char_keys: List[str] = [
    'PRECISION',
    'JOB',
    'ACCURACY',
    'IN.ATOM',
    'CONVERGENCE',
    'XCFUNCTIONAL',
    #'IN.PSP1', # Directly added in `char_dct` in function GeneratePWmatInput.write_etot_input()
    #'IN.PSP2', # Directly added in `char_dct` in function GeneratePWmatInput.write_etot_input()
    #'IN.PSP3', # Directly added in `char_dct` in function GeneratePWmatInput.write_etot_input()
    'VDW',
]

### Type 5. special
special_keys: List[str] = []

### Type 6. string
string_keys: List[str] = [
    'VFF_DETAIL',
    'EGG_DETAIL',
    'N123',
    'NS123',
    'N123L',
    'STRESS_CORR',
    'P123',
    'HSE_DETAIL',
    'RELAX_HSE',
    'IN.OCC',
    'SCF_ITER1_1',
    'IN.A_FIELD',
    'LDAU_PSP1',
    'LDAU_PSP2',
    'MD_INTERACT'
]

### Type 7. list_int
list_int_keys: List[str] = [
    'MP_N123',          # e.g. 'MP_N123 = 3 3 3 0 0 0 0'
    "DOS_DETAIL",       # Set when calculating DOS. Note: kmesh must equal to MP_N123 in the last SCF or NONSCF. e.g. '0 3 3 3'
]

### Type 8. list_float
list_float_keys: List[str] = [
    'SCF_ITER0_1',
    'SCF_ITER0_2',
    'MD_DETAIL',
    "RELAX_DETAIL",             # Set when relax structure
]

# PWmat has 8 types of parameters. (Make variables in `xxx_keys` above to upper, then assign to themselves.)
float_keys = [tmp_key.upper() for tmp_key in float_keys]
int_keys = [tmp_key.upper() for tmp_key in int_keys]
bool_keys = [tmp_key.upper() for tmp_key in bool_keys]
char_keys = [tmp_key.upper() for tmp_key in char_keys]
special_keys = [tmp_key.upper() for tmp_key in special_keys]
string_keys = [tmp_key.upper() for tmp_key in string_keys]
list_int_keys = [tmp_key.upper() for tmp_key in list_int_keys]
list_float_keys = [tmp_key.upper() for tmp_key in list_float_keys]


class GeneratePWmatInput:
    """Generates input files for PWmat."""
    
    # Environment variable for PP paths
    PWMAT_PP_PATH: str = 'PWMAT_PP_PATH'
    
    def __init__(self, job: str = "scf"):
        """
        Set some default values for etot.input file.
        
        Parameters
        ----------
        job : str, default = "scf"
            Represents task type executed by PWmat. You can choose one of 
            'scf', 'nonscf', 'dos', 'relax'
        """
        self.parallel_params: Dict[str, Union[List[int], None]] = {}
        self.float_params: Dict[str, Union[float, None]] = {}
        self.int_params: Dict[str, Union[int, None]] = {}
        self.bool_params: Dict[str, Union[bool, None]] = {}
        self.char_params: Dict[str, Union[str, None]] = {}
        self.special_params: Dict[str, Union[str, None]] = {}
        self.string_params: Dict[str, Union[str, None]] = {}
        self.list_int_params: Dict[str, Union[List[int], None]] = {}
        self.list_float_params: Dict[str, Union[List[float], None]] = {}

        for tmp_key in parallel_keys:
            if (tmp_key == "PARALLEL"):
                self.parallel_params.update({"PARALLEL": [4, 1]})
            else:
                self.parallel_params.update({tmp_key: None})
        ### Type 4. char
        for tmp_key in char_keys:
            if (tmp_key == "JOB"):
                self.char_params.update({"JOB": job.upper()})
            elif (tmp_key == "XCFUNCTIONAL"):
                self.char_params.update({"XCFUNCTIONAL": "PBE"})
            elif (tmp_key == "ACCURACY"):
                if (self.char_params.get("JOB") == "RELAX"):
                    self.char_params.update({"ACCURACY": "HIGH"})
                else:
                    self.char_params.update({"ACCURACY": "NORM"})
            elif (tmp_key == "CONVERGENCE"):
                self.char_params.update({"CONVERGENCE": "EASY"})
            elif (tmp_key == "PRECISION"):
                self.char_params.update({"PRECISION": "AUTO"})
            elif (tmp_key == "IN.ATOM"):
                self.char_params.update({"IN.ATOM": "atom.config"})
            else:
                self.char_params.update({tmp_key: None})
        ### Type 1. float
        for tmp_key in float_keys:
            if (tmp_key == "ECUT"):
                if (self.char_params.get("JOB") == "RELAX"):
                    self.float_params.update({"ECUT": 70})
                else:
                    self.float_params.update({"ECUT": 50})
            elif (tmp_key == "DOS_GAUSSIAN_BROADENING"):    # Set DOS_GAUSSIAN_BROADENING when JOB=DOS
                if (self.char_params.get("JOB") == "DOS"):
                    self.float_params.update({"DOS_GAUSSIAN_BROADENING": 0.05})
                else:
                    self.float_params.update({"DOS_GAUSSIAN_BROADENING": None})
            else:
                self.float_params.update({tmp_key: None})
        ### Type 2. int
        for tmp_key in int_keys:
            if (tmp_key == "SPIN"):
                self.int_params.update({"SPIN": 1})
            elif (tmp_key == "NUM_DOS_GRID"):   # Set NUM_DOS_GRID when JOB=DOS
                if (self.char_params.get("JOB") == "DOS"):
                    self.int_params.update({"NUM_DOS_GRID": 4000})
                else:
                    self.int_params.update({"NUM_DOS_GRID": None})
            else:
                self.int_params.update({tmp_key: None})
        ### Type 3. bool
        for tmp_key in bool_keys:
            if tmp_key in ["OUT.FORCE", "OUT.STRESS", "CHARGE_DECOMP"]:
                self.bool_params.update({tmp_key: True})
            elif (tmp_key == "IN.WG"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"IN.WG": False})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"IN.WG": False})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"IN.WG": True})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"IN.WG": False})
                else:
                    self.bool_params.update({"IN.WG": None})
            elif (tmp_key == "IN.RHO"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"IN.RHO": False})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"IN.RHO": False})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"IN.RHO": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"IN.RHO": False})
                else:
                    self.bool_params.update({"IN.RHO": None})
            elif (tmp_key == "IN.VR"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"IN.VR": False})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"IN.VR": True})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"IN.VR": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"IN.VR": False})
                else:
                    self.bool_params.update({"IN.VR": None})
            elif (tmp_key == "IN.KPT"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"IN.KPT": False})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"IN.KPT": True})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"IN.KPT": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"IN.KPT": False})
                else:
                    self.bool_params.update({"IN.KPT": None})
            elif (tmp_key == "OUT.WG"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"OUT.WG": True})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"OUT.WG": True})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"OUT.WG": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"OUT.WG": True})
                else:
                    self.bool_params.update({"OUT.WG": None})
            elif (tmp_key == "OUT.RHO"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"OUT.RHO": True})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"OUT.RHO": False})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"OUT.RHO": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"OUT.RHO": True})
                else:
                    self.bool_params.update({"OUT.RHO": None})
            elif (tmp_key == "OUT.VR"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"OUT.VR": True})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"OUT.VR": False})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"OUT.VR": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"OUT.VR": True})
                else:
                    self.bool_params.update({"OUT.VR": None})
            elif (tmp_key == "OUT.VATOM"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.bool_params.update({"OUT.VATOM": False})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.bool_params.update({"OUT.VATOM": False})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.bool_params.update({"OUT.VATOM": False})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.bool_params.update({"OUT.VATOM": False})
                else:
                    self.bool_params.update({"OUT.VATOM": None})
            else:
                self.bool_params.update({tmp_key: None})
        ### Type 5. special
        for tmp_key in special_keys:
            self.special_params.update({tmp_key: None})
        ### Type 6. string
        for tmp_key in string_keys:
            self.string_params.update({tmp_key: None})
        ### Type 7. list_int
        for tmp_key in list_int_keys:
            if (tmp_key == "MP_N123"):
                self.list_int_params.update({"MP_N123": [3, 3, 3, 0, 0, 0, 0]})
            elif (tmp_key == "DOS_DETAIL"):     # Set DOS_DETAIL when JOB=DOS
                if (self.char_params.get("JOB") == "DOS"):
                    self.list_int_params.update({"DOS_DETAIL": [0, 3, 3, 3]})
                else:
                    self.list_int_params.update({"DOS_DETAIL": None})
            else:
                self.list_int_params.update({tmp_key: None})
        ### Type 8. list_float
        for tmp_key in list_float_keys:
            if (tmp_key == "SCF_ITER0_1"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.list_float_params.update({"SCF_ITER0_1": [6, 4, 3, 0.0, 0.025, 1]})
                elif (self.char_params.get("JOB") == "NONSCF"):
                    self.list_float_params.update({"SCF_ITER0_1": [50, 4, 3, 0.0, 0.025, 1]})
                elif (self.char_params.get("JOB") == "DOS"):
                    self.list_float_params.update({"SCF_ITER0_1": [50, 4, 3, 0.0, 0.025, 1]})
                elif (self.char_params.get("JOB") == "RELAX"):
                    self.list_float_params.update({"SCF_ITER0_1": [6, 4, 3, 0.0, 0.025, 1]})
                else:
                    self.list_float_params.update({"SCF_ITER0_1": None})
            elif (tmp_key == "SCF_ITER0_2"):
                if (self.char_params.get("JOB") == "SCF"):
                    self.list_float_params.update({"SCF_ITER0_2": [94, 4, 3, 1.0, 0.025, 1]})
                elif(self.char_params.get("JOB") == "RELAX"):
                    self.list_float_params.update({"SCF_ITER0_2": [94, 4, 3, 1.0, 0.025, 1]})
                else:
                    self.list_float_params.update({"SCF_ITER0_2": None})
            elif (tmp_key == "SCF_ITER1_1"):
                if (self.char_params.get("JOB") == "RELAX"):
                    self.list_float_params.update({"SCF_ITER1_1": [40, 4, 3, 1.0, 0.025, 1]})
                else:
                    self.list_float_params.update({"SCF_ITER1_1": None})
            elif (tmp_key == "RELAX_DETAIL"):
                if (self.char_params.get("JOB") == "RELAX"):
                    self.list_float_params.update({"RELAX_DETAIL": [1, 100, 0.01, 1, 0.01]})
                else:
                    self.list_float_params.update({"RELAX_DETAIL": None})
            else:
                self.list_float_params.update({tmp_key: None})

    def create_inputfiles(
        self, 
        atoms: Atoms, 
        directory: str = ".",
        parallel: List[int] = [4, 1],
        **kwargs) -> None:
        """Generate all inputfiles for PWmat calculation tasks."""
        self.add_magmom(atoms=atoms, directory=directory)
        self.copy_pspx(atoms=atoms, directory=directory)
        self.write_etot_input(atoms=atoms, directory=directory, parallel=parallel, **kwargs)
    
    def add_magmom(
        self, 
        atoms: Atoms,
        directory:str = '.') -> None:
        """Add/Modify magnetic moments at the end of atom.config"""        
        atom_config_path = Path(directory) / "atom.config"
        tmp_atom_config_path = Path(directory) / "atom.config_"
        if not os.path.isfile(atom_config_path):
            raise ReadError(f"Cannot read file {atom_config_path}!")
        if os.path.isfile(tmp_atom_config_path):
            os.remove(tmp_atom_config_path)
        
        with open(atom_config_path, 'r') as f, open(tmp_atom_config_path, 'w') as tmp_f:
            for line in f:
                if "MAGNETIC" not in line.upper():
                    tmp_f.write(line)
                else:
                    break
        os.remove(atom_config_path)
        os.rename(tmp_atom_config_path, atom_config_path)

        with open(atom_config_path, "a") as f:
            f.write("MAGNETIC\n")
            for atom in atoms:
                f.write(f"{atom.number} {atom.magmom}\n")
                
    def copy_pspx(self, atoms: Atoms, directory: str = ".") -> None:
        """Helps to copy pseudopotential files to working directory"""
        pwmat_pp_path: str = cfg["PWMAT_PP_PATH"]
        if (self.char_params.get("XCFUNCTIONAL") == "PBE"):
            pwmat_sg15_path = Path(pwmat_pp_path) / "NCPP-SG15-PBE"
            for symbol in atoms.get_chemical_symbols():
                shutil.copy(
                    Path(pwmat_sg15_path) / f"{symbol}.SG15.PBE.UPF",
                    Path(directory))
        else:
            raise NotImplementedError
              
    def write_etot_input(self, atoms: Atoms, directory: str = "./", parallel: List[int] = [4, 1], **kwargs) -> None:
        """Writes the etot.input file."""
        assert (len(parallel) == 2)
        assert ("job" not in kwargs.keys())
        etot_input_params: Dict[str, Any] = {}
        
        parallel_dct: Dict[str, str] = {"PARALLEL": f"{parallel[0]} {parallel[1]}"}
        etot_input_params.update(parallel_dct)
        # Type 1. float params
        self.float_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in float_keys})
        float_dct: Dict[str, str] = dict(
            (key, f"{val:{FLOAT_FORMAT}}") for key, val
            in self.float_params.items()
            if val is not None
        )
        etot_input_params.update(float_dct)
        # Type 2. int params
        self.int_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in int_keys})
        int_dct: Dict[str, Optional[int]] = dict(
            (key, val) for key, val
            in self.int_params.items()
            if val is not None
        )
        etot_input_params.update(int_dct)
        # Type 3. bool params
        self.bool_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in bool_keys})
        bool_dct: Dict[str, str] = dict(
            (key, ('T' if (val is True) else 'F')) for key, val
            in self.bool_params.items()
            if val is not None
        )
        etot_input_params.update(bool_dct)
        # Type 4. char params
        self.char_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in char_keys})
        char_dct: Dict[str, str] = dict(
            (key, val) for key, val
            in self.char_params.items()
            if val is not None
        )
        # Write IN.PSPX
        for tmp_idx, tmp_symbol in enumerate(atoms.get_chemical_symbols()): 
            # Add IN.PSPx = x.SG15.PBE.UPF
            char_dct.update({f"IN.PSP{tmp_idx+1}": f"{tmp_symbol}.SG15.PBE.UPF"})
        etot_input_params.update(char_dct)
        # Type 5. special params
        self.special_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in special_keys})
        special_dct: Dict[str, str] = dict(
            (key, val) for key, val
            in self.special_params.items()
            if val is not None
        )
        etot_input_params.update(special_dct)
        # Type 6. string params
        self.string_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in string_keys})
        string_dct: Dict[str, str] = dict(
            (key, val) for key, val 
            in self.string_params.items()
            if val is not None
        )
        etot_input_params.update(string_dct)
        # Type 7. list_int params
        self.list_int_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in list_int_keys})
        list_int_dct: Dict[str, str] = dict(
            (key, " ".join([str(tmp_item) for tmp_item in val])) for key, val
            in self.list_int_params.items()
            if val is not None
        )
        etot_input_params.update(list_int_dct)
        # Type 8. list_float params
        self.list_float_params.update({k.upper() : v for k, v in kwargs.items() if k.upper() in list_float_keys})
        list_float_dct: Dict[str, str] = dict(
            (key, " ".join([str(tmp_item) for tmp_item in val])) for key, val
            in self.list_float_params.items()
            if val is not None
        )
        etot_input_params.update(list_float_dct)
        ### Output
        write_etot_input(directory=directory, parameters=etot_input_params)
    