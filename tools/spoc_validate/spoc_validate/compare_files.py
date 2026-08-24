"""Stage 2 — Compare Files (optional).

For each obs type's subdirectory (created by stage 1), inventories every
SPOC IODA output file (*_bq_ioda.nc4, or *_bq_ioda_<label>.nc4 for split
obs types) against its matching ncdiag IODA file, and writes:

  - <tag>_qc_inventory.csv    : per-variable QC-based counts/means
  - <tag>_discrepancies.txt   : human-readable summary of any issues

where <tag> is "<obs_type>" for non-split obs types, or
"<obs_type>_<label>" for each split (e.g. "adpupa_sonde", "adpupa_pibal").

Only the ObsValue group and its paired quality-marker group are inspected
(QualityMarker / PreQC / etc, whichever the file actually has) — every
other group (ObsError, MetaData, GSI-specific diagnostic groups, and so
on) is irrelevant to this comparison and is ignored entirely.

For each ObsValue variable that has a matching quality-marker variable,
the inventory records:
  - total observation count for the whole file (once, not per variable)
  - count of observations with quality marker <= 2
  - count of observations with quality marker > 2
  - mean ObsValue where quality marker is 0-2 inclusive

ncdiag files are located via the same experiment.template used for Stage
3's swell hofx setup, rather than a manually pre-populated directory of
links: `models.<model>.ioda_locations_not_in_r2d2` in that template gives
a base directory, and the actual file for a given obs space is found at
<that directory>/<cycle_compact>/geos_atmosphere/<obs_space>.<cycle_compact>.nc4
— exact, per-obs-space, no substring matching or split-label ambiguity
needed (each JEDI obs space has its own distinctly-named ncdiag file).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from . import common, nc_utils
from . import obs_dict as obs_dict_mod

REQUIRED_KEYS = ["test_directory", "obs_dict_file", "experiment_template_file"]

SPOC_INFIX = "_bq_ioda_"
NCDIAG_INFIX = "_ncdiag_ioda_"


def read_ioda_locations_not_in_r2d2(experiment_template_file: Path) -> str:
    """Read models.<model>.ioda_locations_not_in_r2d2 out of a swell
    experiment.yaml/template. Raises a clear error if it's missing, or if
    more than one model component declares a different value (this
    pipeline doesn't know which one you'd want in that case).
    """
    data = yaml.safe_load(Path(experiment_template_file).read_text()) or {}
    models = data.get("models") or {}

    found: Dict[str, str] = {}
    for model_name, model_cfg in models.items():
        if isinstance(model_cfg, dict) and model_cfg.get("ioda_locations_not_in_r2d2"):
            found[model_name] = model_cfg["ioda_locations_not_in_r2d2"]

    if not found:
        raise KeyError(
            f"'ioda_locations_not_in_r2d2' not found under any models.<model> "
            f"in {experiment_template_file}"
        )
    if len(set(found.values())) > 1:
        raise ValueError(
            f"Multiple different ioda_locations_not_in_r2d2 values found across "
            f"model components in {experiment_template_file}: {found} — not sure "
            f"which one to use"
        )
    return next(iter(found.values()))


def _stage_ncdiag_file(entry: "obs_dict_mod.ObsDictEntry", obs_dir: Path,
                        ioda_locations_not_in_r2d2: str, logger) -> Optional[Path]:
    """Locate and copy this obs space's ncdiag file
    (<ioda_locations_not_in_r2d2>/<cycle_compact>/geos_atmosphere/
    <obs_space>.<cycle_compact>.nc4) into obs_dir as
    <obs_space>_ncdiag_ioda.nc4. Returns the staged path, or None (with a
    warning logged) if the cycle time couldn't be parsed or the file
    doesn't exist.
    """
    obs_space = entry.jedi_obs_space
    try:
        _, cycle_compact = common.parse_cycle_from_bufr_path(entry.bufr_path)
    except ValueError as e:
        logger.warning(f"[{obs_space}] {e}")
        return None

    # The ncdiag directory is named for the true cycle time, but the
    # filename's own timestamp is 3 hours behind that (e.g. cycle
    # 20251129T120000Z -> directory .../20251129T120000Z/..., but
    # filename sondes.20251129T090000Z.nc4).
    file_time_compact = common.offset_compact_time(cycle_compact, hours=-3)

    src = (Path(ioda_locations_not_in_r2d2) / cycle_compact / "geos_atmosphere" /
           f"{obs_space}.{file_time_compact}.nc4")
    if not src.is_file():
        logger.warning(f"[{obs_space}] ncdiag file not found at {src}")
        return None

    dest = obs_dir / f"{obs_space}_ncdiag_ioda.nc4"
    common.safe_copy(src, dest, resolve_symlinks=False)
    return dest


def _compare_one(tag: str, spoc_file: Path, ncdiag_file: Path, obs_dir: Path,
                  logger, note: Optional[str] = None) -> Dict[str, Any]:
    """Inventory a single SPOC output file against a single ncdiag file and
    write the CSV + discrepancy txt for it, under the given `tag`.
    """
    spoc_inv = nc_utils.inventory_obs_file(spoc_file)
    ncdiag_inv = nc_utils.inventory_obs_file(ncdiag_file)

    all_vars = sorted(set(spoc_inv["variables"]) | set(ncdiag_inv["variables"]))

    csv_path = obs_dir / f"{tag}_qc_inventory.csv"
    discrepancies = []

    with open(csv_path, "w", newline="") as f:
        f.write(f"# total_obs_spoc={spoc_inv['total_obs']}, "
                f"total_obs_ncdiag={ncdiag_inv['total_obs']}, "
                f"spoc_qc_group={spoc_inv['qc_group_used']}, "
                f"ncdiag_qc_group={ncdiag_inv['qc_group_used']}\n")
        writer = csv.writer(f)
        writer.writerow([
            "variable", "in_spoc", "in_ncdiag",
            "spoc_n_qc_le_2", "spoc_n_qc_gt_2", "spoc_mean_value_qc_le_2",
            "ncdiag_n_qc_le_2", "ncdiag_n_qc_gt_2", "ncdiag_mean_value_qc_le_2",
        ])
        for var in all_vars:
            s = spoc_inv["variables"].get(var)
            n = ncdiag_inv["variables"].get(var)

            s_ok = s is not None and s.get("status") == "ok"
            n_ok = n is not None and n.get("status") == "ok"

            writer.writerow([
                var,
                s is not None,
                n is not None,
                s.get("n_qc_le_2") if s_ok else "",
                s.get("n_qc_gt_2") if s_ok else "",
                s.get("mean_value_qc_le_2") if s_ok else "",
                n.get("n_qc_le_2") if n_ok else "",
                n.get("n_qc_gt_2") if n_ok else "",
                n.get("mean_value_qc_le_2") if n_ok else "",
            ])

            if s is None:
                discrepancies.append(f"Variable '{var}' present in ncdiag ObsValue but MISSING from SPOC ObsValue")
                continue
            if n is None:
                discrepancies.append(f"Variable '{var}' present in SPOC ObsValue but MISSING from ncdiag ObsValue")
                continue
            if s.get("status") == "no_matching_qc_variable":
                discrepancies.append(f"Variable '{var}': no matching quality-marker variable found in SPOC "
                                      f"(looked in group '{s.get('qc_group')}')")
            if n.get("status") == "no_matching_qc_variable":
                discrepancies.append(f"Variable '{var}': no matching quality-marker variable found in ncdiag "
                                      f"(looked in group '{n.get('qc_group')}')")
            if not (s_ok and n_ok):
                continue

            s_total = s["n_qc_le_2"] + s["n_qc_gt_2"]
            n_total = n["n_qc_le_2"] + n["n_qc_gt_2"]
            if s_total > 0 and n_total > 0:
                denom = max(s_total, n_total)
                if abs(s_total - n_total) / denom > 0.01:
                    discrepancies.append(
                        f"Variable '{var}': QC'd observation count differs by >1% "
                        f"(SPOC={s_total} vs ncdiag={n_total})"
                    )

            sm, nm = s.get("mean_value_qc_le_2"), n.get("mean_value_qc_le_2")
            if sm is not None and nm is not None:
                denom = abs(nm) if abs(nm) > 1e-9 else 1.0
                if abs(sm - nm) / denom > 0.01:
                    discrepancies.append(
                        f"Variable '{var}': mean (QC<=2) differs by >1% "
                        f"(SPOC={sm:.6g} vs ncdiag={nm:.6g})"
                    )

    txt_path = obs_dir / f"{tag}_discrepancies.txt"
    with open(txt_path, "w") as f:
        f.write(f"Discrepancy report: {tag}\n")
        f.write(f"SPOC file:   {spoc_file}\n")
        f.write(f"ncdiag file: {ncdiag_file}\n")
        f.write(f"SPOC total obs:   {spoc_inv['total_obs']} (QC group: {spoc_inv['qc_group_used']})\n")
        f.write(f"ncdiag total obs: {ncdiag_inv['total_obs']} (QC group: {ncdiag_inv['qc_group_used']})\n")
        if spoc_inv["total_obs"] is not None and ncdiag_inv["total_obs"] is not None:
            denom = max(spoc_inv["total_obs"], ncdiag_inv["total_obs"], 1)
            if abs(spoc_inv["total_obs"] - ncdiag_inv["total_obs"]) / denom > 0.01:
                discrepancies.insert(0, f"Total obs count differs by >1% "
                                         f"(SPOC={spoc_inv['total_obs']} vs ncdiag={ncdiag_inv['total_obs']})")
        if note:
            f.write(f"\nNOTE: {note}\n")
        f.write("\n")
        if discrepancies:
            f.write(f"{len(discrepancies)} issue(s) found:\n")
            for d in discrepancies:
                f.write(f"  - {d}\n")
        else:
            f.write("No discrepancies found.\n")

    logger.info(f"[{tag}] wrote {csv_path.name} and {txt_path.name} "
                f"({len(discrepancies)} discrepancies)")

    return {
        "status": "ok",
        "spoc_file": str(spoc_file),
        "ncdiag_file": str(ncdiag_file),
        "csv": str(csv_path),
        "discrepancies_txt": str(txt_path),
        "n_discrepancies": len(discrepancies),
        "spoc_total_obs": spoc_inv["total_obs"],
        "ncdiag_total_obs": ncdiag_inv["total_obs"],
        "note": note,
    }


def compare_obs_type(obs_type: str, obs_dir: Path, logger) -> Dict[str, Dict[str, Any]]:
    """Inventory every SPOC split output file for this obs type against its
    matching (or best-available) ncdiag file. Returns a dict keyed by split
    label ("default" if the obs type has no splits).
    """
    spoc_files = sorted(obs_dir.glob(f"{obs_type}{SPOC_INFIX}*.nc4")) or \
                 sorted(obs_dir.glob(f"{obs_type}_bq_ioda.nc4"))
    ncdiag_files = sorted(obs_dir.glob(f"{obs_type}{NCDIAG_INFIX}*.nc4")) or \
                   sorted(obs_dir.glob(f"{obs_type}_ncdiag_ioda.nc4"))

    if not spoc_files:
        logger.warning(f"[{obs_type}] no SPOC IODA output found; skipping")
        return {"default": {"status": "missing_spoc_file"}}
    if not ncdiag_files:
        logger.warning(f"[{obs_type}] no ncdiag IODA file found; skipping")
        return {"default": {"status": "missing_ncdiag_file"}}

    ncdiag_by_label: Dict[str, Path] = {}
    shared_ncdiag: Optional[Path] = None
    for f in ncdiag_files:
        label = common.extract_split_label(f.name, obs_type, NCDIAG_INFIX)
        if label:
            ncdiag_by_label[label] = f
        else:
            shared_ncdiag = f
    if shared_ncdiag is None and len(ncdiag_files) == 1 and not ncdiag_by_label:
        shared_ncdiag = ncdiag_files[0]

    results: Dict[str, Dict[str, Any]] = {}

    for spoc_file in spoc_files:
        label = common.extract_split_label(spoc_file.name, obs_type, SPOC_INFIX)
        tag = f"{obs_type}_{label}" if label else obs_type
        key = label or "default"

        note = None
        if label and label in ncdiag_by_label:
            ncdiag_file = ncdiag_by_label[label]
        elif shared_ncdiag is not None:
            ncdiag_file = shared_ncdiag
            if label:
                note = (f"No split-specific ncdiag file found for split '{label}'; "
                        f"compared against the single shared ncdiag file "
                        f"{shared_ncdiag.name} instead. QC-count/mean comparisons "
                        f"may not be apples-to-apples if the ncdiag file covers "
                        f"records outside this split.")
                logger.warning(f"[{tag}] {note}")
        else:
            logger.warning(
                f"[{tag}] could not determine which ncdiag file to compare against "
                f"(multiple ncdiag files present, none labeled to match '{label}'); skipping"
            )
            results[key] = {"status": "ambiguous_ncdiag_match"}
            continue

        results[key] = _compare_one(tag, spoc_file, ncdiag_file, obs_dir, logger, note)

    return results


def run(config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    common.require_keys(config, REQUIRED_KEYS, context="compare_files config")

    test_dir = Path(config["test_directory"])
    logger = common.get_logger("compare_files", test_dir / "compare_files.log")
    logger.info("=== Stage 2: Compare Files ===")

    entries = obs_dict_mod.load_obs_dict(config["obs_dict_file"])
    ioda_locations_not_in_r2d2 = read_ioda_locations_not_in_r2d2(config["experiment_template_file"])
    logger.info(f"Using ioda_locations_not_in_r2d2 = {ioda_locations_not_in_r2d2}")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in entries:
        obs_type = entry.jedi_obs_space
        obs_dir = test_dir / obs_type
        if not obs_dir.is_dir():
            logger.warning(f"[{obs_type}] subdirectory {obs_dir} does not exist; skipping")
            results[obs_type] = {"default": {"status": "missing_subdir"}}
            continue

        if _stage_ncdiag_file(entry, obs_dir, ioda_locations_not_in_r2d2, logger) is None:
            results[obs_type] = {"default": {"status": "missing_ncdiag_file"}}
            continue

        results[obs_type] = compare_obs_type(obs_type, obs_dir, logger)

    n_ok = sum(
        1 for splits in results.values() for r in splits.values() if r.get("status") == "ok"
    )
    n_total = sum(len(splits) for splits in results.values())
    logger.info(f"=== Stage 2 complete: {n_ok}/{n_total} comparisons written OK ===")
    return results
