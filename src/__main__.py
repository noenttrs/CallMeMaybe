"""Allow running the program with ``python -m src``."""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
