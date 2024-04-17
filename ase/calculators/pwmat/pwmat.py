"""Define a calculator for PWmat"""
from __future__ import annotations
import io
import os
import subprocess
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.pwmat import GeneratePWmatInput
from ase.atoms import Atoms
from ase.config import cfg
from ase.io import read

__author__ = "Hanyu Liu"
__email__ = "domainofbuaa@gmail.com"
__date__ = "2024-1-17"


class PWmat(GeneratePWmatInput, Calculator):
    """ASE interface for the PWmat, with the Calculator interface.
    
    Parameters:
    
    atoms: object
        Attach an atoms object to the calculator.

    job: str
        The task type of PWmat to execute
    
    label: str
        Prefix for the output file, Default is 'pwmat'.

    directory: str
        Set the working directory. Is prepended to ``label``.

    restart: str or bool
        Sets a label for the directory to load files from.
        if :code:`restart=True`, the working directory from
        ``directory`` is used.

    txt: bool, None, str or writable object
        - If txt is None, output stream will be supressed

        - If txt is '-' the output will be sent through stdout

        - If txt is a string a file will be opened,\
            and the output will be sent to that file.

        - Finally, txt can also be a an output stream,\
            which has a 'write' attribute.

        Default is 'pwmat.out'
    
        - Requsite:
            $ export PWMAT_PP_PATH=<your_path_to_pp>/PseudoPotential
            $ export ASE_PWMAT_COMMAND="mpirun -np 1 PWmat| tee output"
        - Examples:
        >>> from ase.io import read
        >>> from ase.calculator.pwmat import pwmat
        >>> atoms=read("./atom.config")
        >>> pwmat=PWmat(atoms=atoms, job='scf', parallel=[4,1], directory=".")
        >>> pwmat.calculate()  # launch the task.
    
    command: str
        Custom instructions on how to execute PWmat. Has priority over
        environment variables.
    """
    # Environment commands
    env_commands = ('ASE_PWMAT_COMMAND', 'PWMAT_COMMAND', 'PWMAT_SCRIPT')

    def __init__(self,
                 atoms: Optional[Union[Atoms, None]] = None,
                 job: str = "scf",
                 parallel: Optional[List[int]] = None,
                 restart: Optional[str] = None,
                 directory: str = ".",
                 label: str = "pwmat",
                 ignore_bad_restart_file=Calculator._deprecated,
                 command: Optional[str] = None,
                 txt: Optional[str] = 'pwmat.out',
                 **kwargs):
        if parallel is None:
            parallel = [4, 1]
        assert len(parallel) == 2
        self.parallel: List[int] = parallel
        self.atoms = atoms
        self.results: Dict[str, Any] = {}
        self.kwargs = kwargs
        # Initialize parameter dictionaries
        GeneratePWmatInput.__init__(self, job=job)
        self._store_param_state()   # Initialize an empty parameter state

        # Store calculator from vasprun.xml here - None => uninitialized
        self._xml_calc: Optional[SinglePointDFTCalculator] = None

        # Set directory and label
        '''
        Examples to explain `directory`, `label`
        
        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')
        * label=None: (directory='.', prefix=None)
        '''
        self.directory: str = directory
        if "/" in label:
            if self.directory != ".":
                msg = (
                    'Directory redundantly specified though directory='
                    f'"{self.directory}" and label="{label}".  Please omit '
                    '"/" in label.'
                )
                raise ValueError(msg)
            self.label: str = label
        else:
            self.prefix: str = label # The label should only contain the prefix

        if isinstance(restart, bool):
            restart = self.label if restart is True else None

        Calculator.__init__(
            self,
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            label=self.label,
            atoms=atoms,
            **kwargs)
        self.command: Optional[str] = command
        self._txt = None
        self.txt = txt  # Set the output txt stream
        self.version = None
        self.atoms = atoms

    def write_input(self, atoms: Atoms):
        """Write all inputfiles"""
        # Create the folders where we write the files, if we aren't in the 
        # current working directory
        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self.initialize(atoms)
        self.create_inputfiles(atoms=atoms,
                               directory=self.directory,
                               parallel=self.parallel,
                               kwargs=self.kwargs)

    def initialize(self, atoms: Atoms):
        self.atoms = atoms
        #self.natoms = len(atoms)

    def calculate(self, 
                  atoms: Optional[Atoms] = None, 
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
        """Do a PWmat calculation in the specified directory.
        
        This will generate the necessary PWmat input files, and then
        execute PWmat. After execution, the energy, forces. etc. are read
        from the PWmat output files.
        """
        command: str = self.make_command(self.command)
        
        if atoms is not None:
            self.atoms = atoms.copy()
        self.write_input(self.atoms)
        
        with self._txt_outstream() as out:
            errorcode = self._run(command=command,
                                  out=out,
                                  directory=self.directory)
        
        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d}'.format(
                    self.name, Path(self.directory).resolve(), errorcode))
        # Read results from calculation: Temporarily not implemented.
                
    def _run(self, 
             command: str, 
             out: Optional[io.TextIOWrapper]=None, 
             directory: Optional[str]=None) -> int:
        """Method to explicitly execute PWmat"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory
        errorcode: int = subprocess.call(command,
                                         shell=True,
                                         stdout=out,
                                         cwd=directory)
        return errorcode

    def make_command(self, command:Union[str, None] = None) -> str:
        """Return command if one is passed, otherwise try to find 
        ASE_PWMAT_COMMAND, PWMAT_COMMAND or PWMAT_SCRIPT.
        If none are set, a CalculatorSetUpError is raised"""
        cmd: str = ""
        if command:
            cmd = command
        else:
            # Search for the environment commands
            for env in self.env_commands:
                if env in cfg:
                    cmd = cfg[env].replace("PREFIX", self.prefix)
        return cmd

    def read(self, label: Union[str, None] = None):
        """Read results from PWmat output files.
        
        Files which are read: 
            - final.config
            - REPORT (No implementation now)
            
        Raises ReadError if they are not found"""
        if label is None:
            label = self.label
        Calculator.read(self, label)

        # If we restart, self.parameetrs isn't initialized
        if self.parameters is None:
            self.parameters = self.get_default_parameters()

        # Check for existence of the necessary output files
        for f in ["REPORT"]:
            file = self._indir(filename=f)
            if not file.is_file():
                raise calculator.ReadError(
                    f'PWmat outputfile {file} was not found')
        # Read atoms
        self.atoms = self.read_pwmat_atoms(filename=self._indir("final.config"))
        # Read parameters. No implementation now.
        # Read results from the calculation. No implementation now.

    def _indir(self, filename: str):
        "Prepend current directory to filename"
        return Path(self.directory) / filename

    def _store_param_state(self) -> None:
        """Store current parameter state."""
        self.param_state: Dict[str, Dict[str, Any]] = dict(
            parallel_params=self.parallel_params.copy(),
            float_params=self.float_params.copy(),
            int_params=self.int_params.copy(),
            bool_params=self.bool_params.copy(),
            char_params=self.char_params.copy(),
            special_params=self.special_params.copy(),
            string_params=self.string_params.copy(),
            list_int_params=self.list_int_params.copy(),
            list_float_params=self.list_float_params.copy())

    @contextmanager
    def _txt_outstream(self):
        """Custom function for opening a text output stream. Uses self.txt
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes
        the new stream again when exiting.

        Examples:
        # Pass a string
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'

        # Use an existing stream
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout
        """

        txt = self.txt
        open_and_close = False  # Do we open the file?

        if txt is None:
            # Suppress stdout
            out = subprocess.DEVNULL
        else:
            if isinstance(txt, str):
                if txt == '-':
                    # subprocess.call redirects this to stdout
                    out = None
                else:
                    # Open the file in the work directory
                    txt = self._indir(txt)
                    # We wait with opening the file, until we are inside the
                    # try/finally
                    open_and_close = True
            elif hasattr(txt, 'write'):
                out = txt
            else:
                raise RuntimeError(
                    f'txt should be a string or an I/O stream, got {txt}')

        try:
            if open_and_close:
                out = open(txt, 'w', encoding='utf-8')
            yield out
        finally:
            if open_and_close:
                out.close()

    def read_pwmat_atoms(self, filename: str):
        """Read the atoms from file located in the PWmat
        working directory. Normally called final.config."""
        return read(filename)
    