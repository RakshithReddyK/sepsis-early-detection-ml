"""Unified CLI: `sepsis-ml <subcommand> [options]`.

Subcommands:
  train      Train + cross-validate the sepsis model, save model + metrics.
  data-prep  Generate a reconstructed synthetic dataset (see data.py).
"""
from __future__ import annotations

import argparse
import sys

from . import data as data_mod
from . import train as train_mod


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="sepsis-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train the sepsis model.")
    subparsers.add_parser("data-prep", help="Generate a reconstructed synthetic dataset.")

    # Only peel off the subcommand name here; each subcommand parses its
    # own remaining args with its existing argparse parser.
    if not argv or argv[0] not in {"train", "data-prep"}:
        parser.parse_args(argv)  # will error with help text
        return

    command, rest = argv[0], argv[1:]
    if command == "train":
        train_mod._cli(rest)
    elif command == "data-prep":
        data_mod._cli(rest)


if __name__ == "__main__":
    main()
