from windtunnel import __version__
from windtunnel.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_runs_without_command() -> None:
    assert main([]) == 0
