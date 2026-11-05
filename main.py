
"""Command-line entry point for the data recovery project."""

import argparse
import sys
from pathlib import Path

from src.pipeline import ProcessingRun


def main() -> int:
    """Run the data recovery pipeline using the supplied configuration."""
    parser = argparse.ArgumentParser(
        description="Four-format data recovery and validation."
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the project configuration JSON file.",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return 1

    processing_run = ProcessingRun(config_path)
    success = processing_run.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
