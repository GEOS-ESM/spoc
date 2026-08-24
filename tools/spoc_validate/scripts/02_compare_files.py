#!/usr/bin/env python
"""CLI entry point for Stage 2 (optional): Compare Files.

Reuses the same config as stage 1 (needs test_directory, obs_dict_file,
experiment_template_file).

Usage:
    python 02_compare_files.py --config ../config/grab_convert_config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spoc_validate import common, compare_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                         help="Path to grab_convert_config.yaml (stage 1 config, reused here)")
    args = parser.parse_args()

    config = common.load_config(args.config)
    results = compare_files.run(config)

    summary_path = Path(config["test_directory"]) / "stage2_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nStage 2 summary written to {summary_path}")

    n_failed = sum(
        1 for splits in results.values() for r in splits.values() if r.get("status") != "ok"
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
