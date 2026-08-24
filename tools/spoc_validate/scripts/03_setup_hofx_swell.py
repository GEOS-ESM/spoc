#!/usr/bin/env python
"""CLI entry point for Stage 3: Set up hofx tests via swell.

Usage:
    python 03_setup_hofx_swell.py --config ../config/hofx_swell_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoc_validate import common, hofx_swell  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to hofx_swell_config.yaml")
    args = parser.parse_args()

    config = common.load_config(args.config)
    results = hofx_swell.run(config)

    summary_path = Path(config["test_directory"]) / "stage3_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nStage 3 summary written to {summary_path}")

    n_failed = sum(
        1 for splits in results.values() for r in splits.values() if r.get("status") != "ok"
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
