from ase.calculators.vasp.vasp_template import VaspTemplate
from unittest.mock import patch


@patch("ase.io.vasp_parsers.vasp_structure_io.write_vasp_structure", autospec = True)
@patch("ase.io.vasp_parsers.kpoints_writer.write_kpoints", autospec=True)
@patch("ase.io.vasp_parsers.incar_writer.write_incar", autospec=True)
def test_write_input(write_incar,write_kpoints,write_poscar):
    template = VaspTemplate()
    parameters = {"incar": "foo" , "kpoints": "bar"}
    template.write_input("directory", "atoms", parameters, "properties")
    write_incar.assert_called_once_with(parameters["incar"])
    write_kpoints.assert_called_once_with(parameters["kpoints"]) 
    write_poscar.assert_called_once_with("POSCAR","atoms")


