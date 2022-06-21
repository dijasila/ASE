"""This module defines an ASE interface to RESCU+."""

from ase import io
from ase.calculators.calculator import FileIOCalculator


class Rescuplus(FileIOCalculator):
    """ASE interface for RESCU+.

    You may specify the RESCU+ command with the ``command`` keyword of ``__init__`` or
    set the following environment variable as follows::

        export ASE_RESCUPLUS_COMMAND="mpiexec -n 1 rescuplus_scf -i PREFIX.rsi > resculog.out && cp rescuplus_scf_out.json PREFIX.rso"

    All options for ``rescuplus_scf`` may be passed via the dict input_data, as
    in the ``RESCUPy`` module.

    Accepts all the options for ``rescuplus_scf`` as given in the RESCU+ docs,
    plus some additional options:

    pseudopotentials: list
        A list of dictionaries, one for each atomic species, e.g.
        [{'label':'Ga', 'path':'Ga_AtomicData.mat'},
        {'label':'As', 'path':'As_AtomicData.mat'}].
    kpts: array
        List of 3 integers giving the dimensions of a Monkhorst-Pack grid.
        If ``kpts`` is set to ``None``, only the Γ-point will be included.

    .. note::
        Set ``forces_return=True`` and ``stress_return=True`` to calculate forces
        and stresses.
    """

    implemented_properties = ["energy", "forces", "stress"]
    command = "rescuplus_scf -i PREFIX.rsi > PREFIX.rso"
    _deprecated = object()

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=_deprecated,
        label="rescuplus",
        atoms=None,
        **kwargs
    ):
        FileIOCalculator.__init__(
            self, restart, ignore_bad_restart_file, label, atoms, **kwargs
        )
        self.calc = None

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        io.write(self.label + ".rsi", atoms, **self.parameters)

    def read_results(self):
        output, rscobj = io.read(self.label + ".rso")
        self.calc = output.calc
        self.results = output.calc.results
        self.nbands = rscobj.solver.eig.get_number_of_bands()
        self.ngrids = rscobj.system.cell.grid
        self.xc_func = rscobj.system.xc.functional_names

    def get_bz_k_points(self):
        return self.calc.get_bz_k_points()

    def get_effective_potential(self, spin=0, pad=True):
        raise NotImplementedError

    def get_eigenvalues(self, **kwargs):
        return self.calc.get_eigenvalues(**kwargs)

    def get_fermi_level(self):
        return self.calc.get_fermi_level()

    def get_ibz_k_points(self):
        return self.calc.get_ibz_k_points()

    def get_k_point_weights(self):
        return self.calc.get_k_point_weights()

    def get_magnetic_moment(self, atoms=None):
        raise NotImplementedError

    def get_number_of_bands(self):
        return self.nbands

    def get_number_of_grid_points(self):
        return self.ngrids

    def get_number_of_spins(self):
        return self.calc.get_number_of_spins()

    def get_occupation_numbers(self, kpt=0, spin=0):
        raise NotImplementedError

    def get_pseudo_density(self):
        raise NotImplementedError

    def get_pseudo_wavefunction(self, band=0, kpt=0, spin=0, broadcast=True, pad=True):
        raise NotImplementedError

    def get_spin_polarized(self):
        return False

    def get_xc_functional(self):
        return self.xc_func

    def get_wannier_localization_matrix(
        self, nbands, dirG, kpoint, nextkpoint, G_I, spin
    ):
        raise NotImplementedError

    def initial_wannier(
        self, initialwannier, kpointgrid, fixedstates, edf, spin, nbands
    ):
        raise NotImplementedError
