"""Command-line entry point. Subcommands are added stage by stage."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from windtunnel import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="windtunnel", description=__doc__)
    parser.add_argument("--version", action="version", version=f"windtunnel {__version__}")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
