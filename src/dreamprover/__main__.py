"""Allow ``python -m dreamprover`` to use the installed command-line interface."""

from dreamprover.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
