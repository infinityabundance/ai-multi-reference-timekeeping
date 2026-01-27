"""CLI wrapper for the C++ aimrtd daemon."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AIMRT C++ daemon")
    parser.add_argument("--config", required=True, help="Path to TOML configuration file")
    parser.add_argument("--no-chrony", action="store_true", help="Disable Chrony SHM output")
    args = parser.parse_args()

    command = ["aimrtd", "--config", args.config]
    if args.no_chrony:
        command.append("--no-chrony")

    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
