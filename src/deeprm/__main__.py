"""
deeprm package main.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())  # the exit status of `python -m deeprm` is the same as that of `deeprm`
