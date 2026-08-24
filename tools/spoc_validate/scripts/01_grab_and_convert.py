#!/usr/bin/env python
"""CLI entry point for Stage 1: Grab Tag and Convert.

Usage:
    python 01_grab_and_convert.py --config ../config/grab_convert_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoc_validate import common, grab_and_convert  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to grab_convert_config.yaml")
    args = parser.parse_args()

    config = common.load_config(args.config)
    results = grab_and_convert.run(config)

    summary_path = Path(config["test_directory"]) / "stage1_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nStage 1 summary written to {summary_path}")

    n_failed = sum(1 for r in results.values() if r.get("status") != "ok")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
