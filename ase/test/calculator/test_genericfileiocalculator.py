"""Test suite for ase.calculators.GenericFileIOCalculator"""

from subprocess import TimeoutExpired

import pytest
from ase.calculators.genericfileio import BaseProfile, GenericFileIOCalculator
from ase.config import Config, cfg


@pytest.mark.parametrize(
    "calculator_kwargs, result_command",
    [
        ({"parallel": False}, ["dummy.x"]),
        (
            {
                "parallel": False,
                "parallel_info": {"-np": 4, "--oversubscribe": True}
            },
            ["dummy.x"],
        ),
        (
            {
                "parallel": True,
                "parallel_info": {"-np": 4, "--oversubscribe": False}
            },
            ["mpirun", "-np", "4", "dummy.x"],
        ),
        ({"parallel": True}, ["mpirun", "dummy.x"]),
        (
            {
                "parallel": True,
                "parallel_info": {"-np": 4, "--oversubscribe": True}
            },
            ["mpirun", "-np", "4", "--oversubscribe", "dummy.x"],
        ),
        (
            {"parallel": True, "parallel_info": {"nprocs": 4}},
            ["mpirun", "-np", "4", "dummy.x"],
        ),
    ],
)
def test_run_command(
        tmp_path, dummy_template, calculator_kwargs, result_command,
        monkeypatch,
):
    """A test for the command creator from the config file"""

    mock_config = Config()
    mock_config.parser.update({
        "parallel": {"binary": "mpirun", "nprocs_kwarg_trans": "-np"},
        "dummy": {
            "exc": "dummy.x",
        },
    })

    monkeypatch.setattr(cfg, 'parser', mock_config.parser)
    calc = GenericFileIOCalculator(
        template=dummy_template,
        profile=None,
        directory=tmp_path,
        **calculator_kwargs
    )
    assert calc.profile.get_command(inputfile="") == result_command


def test_timeout(tmp_path):
    """A test for the timeout handling"""

    class DummyProfile(BaseProfile):
        def __init__(self, binary, **kwargs):
            super().__init__(**kwargs)
            self.binary = binary

        def get_calculator_command(self, inputfile):
            return [self.binary, str(inputfile)]

        def version(self):
            pass

    profile = DummyProfile(binary="sleep")
    profile.run(tmp_path, "0.01", "dummy", timeout=0.1)

    with pytest.raises(TimeoutExpired):
        profile.run(tmp_path, "0.1", "dummy", timeout=0.01)
