"""Command-line entry point for proving and lemma-library experiments."""

import argparse
import importlib


_COMMANDS = {
    "prove": ("dreamprover.prover.run", "Run recursive proving on a configured dataset"),
    "train": ("dreamprover.pipeline", "Learn a lemma library through wake/sleep cycles"),
    "evaluate": ("dreamprover.pipeline", "Evaluate empty and learned libraries on held-out problems"),
    "assess": ("dreamprover.learning.assessment", "Compare library-assisted retries on unsolved training targets"),
    "monitor": ("dreamprover.runtime.monitor", "Inspect a running experiment without model calls"),
    "report": ("dreamprover.runtime.report", "Write a report from saved experiment checkpoints"),
    "doctor": ("dreamprover.doctor", "Check configuration, dependencies, and optional services"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, (_, description) in _COMMANDS.items():
        commands.add_parser(command, help=description, add_help=False)
    args, command_argv = parser.parse_known_args(argv)
    module = importlib.import_module(_COMMANDS[args.command][0])
    if args.command in {"train", "evaluate"}:
        phase = "training" if args.command == "train" else "evaluation"
        result = module.command_line(phase, command_argv)
    else:
        result = module.main(command_argv)
    return result if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
