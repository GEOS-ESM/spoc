"""Stage 1 — Grab Tag and Convert.

Driven by Obs_dict.txt (see obs_dict.py) rather than a scripts_to_test list
+ a directory of pre-populated bufr symlinks. Each Obs_dict.txt line maps
one JEDI obs space to a SPOC script/yaml basename and a full bufr file
path. Multiple obs spaces can share the same (script, bufr) pair — that's
the normal case when a conversion script splits its output across several
obs spaces (e.g. prepbufr_adpupa.py -> 'sonde' + 'pibal') — so the
conversion for a given (script, bufr) pair only runs once, and its
output(s) get distributed out to each obs space that references it.

  A. Make the test directory (with a subdirectory per JEDI obs space)
  B. Group Obs_dict entries by (script_prefix, bufr_path); for each group,
     grab the '<script_prefix>.py' conversion script and
     '<script_prefix>.yaml' config yaml from the given SPOC tag/branch by
     their exact names, plus any shared '*obs_builder.py' helper modules
     — all copied flat into the same working directory as the script and
     yaml (not PYTHONPATH), since `python script.py` already puts the
     script's own directory on sys.path, and some SPOC scripts locate
     their own yaml config relative to cwd in a way that PYTHONPATH
     manipulation can break
  C. Copy the bufr input for that group
  D. Run the bufr->IODA conversion script from within that working
     directory
  E. Tag the output netCDF(s) with SPOC-tag provenance metadata
  F. Distribute each group's output out to every obs space that
     references it, by fuzzy-matching the JEDI obs space name against the
     split labels the conversion actually produced

ncdiag comparison files are no longer handled here — see compare_files.py
(Stage 2), which locates them itself via the swell experiment.template's
ioda_locations_not_in_r2d2 setting.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import common, nc_utils, obs_dict as obs_dict_mod
from . import repo_utils

SPOC_INFIX = "_bq_ioda_"

REQUIRED_KEYS = ["obs_dict_file", "test_directory"]


def _resolve_repo_source(config: Dict[str, Any]) -> tuple[str, str]:
    """Pull the SPOC source to test out of the config. Exactly one of
    'repo_tag' / 'repo_branch' / 'repo_local_path' must be set; returns
    (ref, ref_type) where ref_type is 'tag', 'branch', or 'local'.
    """
    repo_tag = config.get("repo_tag")
    repo_branch = config.get("repo_branch")
    repo_local_path = config.get("repo_local_path")
    set_count = sum(bool(x) for x in (repo_tag, repo_branch, repo_local_path))
    if set_count != 1:
        raise KeyError(
            "Config must set exactly one of 'repo_tag', 'repo_branch', or "
            f"'repo_local_path' (got repo_tag={repo_tag!r}, "
            f"repo_branch={repo_branch!r}, repo_local_path={repo_local_path!r})"
        )
    if repo_tag:
        return repo_tag, "tag"
    if repo_branch:
        return repo_branch, "branch"
    return repo_local_path, "local"


def _sanitize_for_dirname(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _assign_outputs_for_group(group_entries: List["obs_dict_mod.ObsDictEntry"],
                               output_files: List[Path], conv_key: str) -> Dict[str, Optional[Path]]:
    """Assign each obs space in a conversion group to one of the group's
    output files, considering the whole group at once (see
    common.assign_labels) rather than matching each obs space
    independently — resolves cases like 'sfc' vs 'sfcship' both
    fuzzy-matching the same label if matched in isolation.
    """
    obs_spaces = [e.jedi_obs_space for e in group_entries]

    if len(output_files) == 1 and common.extract_split_label(output_files[0].name, conv_key, SPOC_INFIX) is None:
        return {obs_space: output_files[0] for obs_space in obs_spaces}

    label_to_file: Dict[str, Path] = {}
    for f in output_files:
        label = common.extract_split_label(f.name, conv_key, SPOC_INFIX)
        if label:
            label_to_file[label] = f

    label_assignment = common.assign_labels(obs_spaces, list(label_to_file.keys()))
    return {obs_space: label_to_file.get(label_assignment[obs_space]) for obs_space in obs_spaces}


def _clean_previous_run(test_dir: Path) -> "list[str]":
    """Remove everything Stage 1 owns from a previous run against this same
    test_directory — every top-level entry except the ones explicitly
    preserved below — so stale data (an obs space removed from
    Obs_dict.txt, leftover split files from a since-changed conversion
    script, an old SPOC_VERSION.txt, etc.) can't linger and confuse
    results. Done before the logger is set up so the old log file itself
    gets cleared too, rather than silently appended to. Returns the names
    removed, for the fresh logger to report once it exists.

    Preserved (not Stage 1's to manage):
      - hofx_tests/, _swell_overrides/  (Stage 3 / hofx_swell.py)
      - .spoc_repo_cache/               (meant to persist across runs)
    """
    preserve = {"hofx_tests", "_swell_overrides", ".spoc_repo_cache"}
    if not test_dir.is_dir():
        return []
    removed = []
    for child in sorted(test_dir.iterdir()):
        if child.name in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child.name)
    return removed


def run(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run stage 1 for every JEDI obs space in the Obs_dict.txt file.

    Returns a per-obs-space result dict, e.g.:
        {"sondes": {"status": "ok", "spoc_output": "...", "ncdiag_input": "..."}}
    """
    common.require_keys(config, REQUIRED_KEYS, context="grab_and_convert config")

    test_dir = common.ensure_abs_dir(config["test_directory"])
    removed = _clean_previous_run(test_dir)

    logger = common.get_logger("grab_and_convert", test_dir / "grab_and_convert.log")
    if removed:
        logger.info(f"Cleaned up {len(removed)} item(s) from a previous run in "
                    f"{test_dir}: {removed}")

    entries = obs_dict_mod.load_obs_dict(config["obs_dict_file"])
    groups = obs_dict_mod.conversion_groups(entries)
    logger.info(f"Loaded {len(entries)} obs space(s) from {config['obs_dict_file']}, "
                f"grouped into {len(groups)} conversion job(s)")

    ref, ref_type = _resolve_repo_source(config)

    logger.info(f"=== Stage 1: Grab Tag and Convert ({ref_type}={ref}) ===")

    local_repo_status = None
    if ref_type == "local":
        repo_root = Path(ref)
        if not repo_root.is_dir():
            raise FileNotFoundError(f"repo_local_path does not exist: {repo_root}")
        if not (repo_root / repo_utils.CONFIG_SUBDIR).is_dir() or \
           not (repo_root / repo_utils.SCRIPTS_SUBDIR).is_dir():
            raise FileNotFoundError(
                f"repo_local_path doesn't look like a SPOC checkout (missing "
                f"{repo_utils.CONFIG_SUBDIR} and/or {repo_utils.SCRIPTS_SUBDIR} "
                f"under {repo_root})"
            )
        local_repo_status = repo_utils.get_local_repo_status(repo_root)
        if local_repo_status:
            logger.info(f"  Local repo git status: {local_repo_status}")
    else:
        cache_dir = common.ensure_dir(config.get("repo_cache_directory", test_dir / ".spoc_repo_cache"))
        repo_root = repo_utils.checkout_spoc_ref(ref, cache_dir, ref_type=ref_type, logger=logger)

    config_dir = repo_root / repo_utils.CONFIG_SUBDIR
    scripts_dir = repo_root / repo_utils.SCRIPTS_SUBDIR

    version_stamp = (
        f"SPOC repo ref:  {ref}\n"
        f"Ref type:       {ref_type}\n"
        + (f"Local repo git status: {local_repo_status}\n" if local_repo_status else "")
        + f"Grab/convert run (UTC): {dt.datetime.utcnow().isoformat()}Z\n"
    )
    (test_dir / "SPOC_VERSION.txt").write_text(version_stamp)

    obs_builder_modules = repo_utils.find_obs_builder_modules(scripts_dir)
    if obs_builder_modules:
        logger.info(f"Found {len(obs_builder_modules)} shared obs_builder module(s): "
                    f"{[f.name for f in obs_builder_modules]}")

    results: Dict[str, Dict[str, Any]] = {}
    conversions_root = common.ensure_dir(test_dir / "_conversions")

    for (script_prefix, bufr_path), group_entries in groups.items():
        obs_spaces = [e.jedi_obs_space for e in group_entries]
        logger.info(f"--- conversion job: {script_prefix} ({', '.join(obs_spaces)}) ---")

        work_dir_name = f"{script_prefix}__{_sanitize_for_dirname(Path(bufr_path).name)}"
        work_dir = common.ensure_dir(conversions_root / work_dir_name)

        group_failure: Optional[Dict[str, Any]] = None

        # --- B. config yaml + conversion script, matched by exact name ---
        matched_yaml = repo_utils.find_exact_file(script_prefix, config_dir, ".yaml")
        matched_script = repo_utils.find_exact_file(script_prefix, scripts_dir, ".py")
        if matched_yaml is None or matched_script is None:
            logger.warning(f"  Could not find '{script_prefix}.yaml' and/or "
                            f"'{script_prefix}.py' in the SPOC repo")
            group_failure = {"status": "missing_config_or_script"}
        else:
            local_yaml = common.safe_copy(matched_yaml, work_dir / matched_yaml.name)
            local_script = common.safe_copy(matched_script, work_dir / matched_script.name)

            # obs_builder helper modules are copied flat into work_dir
            # (same directory as the script and yaml), NOT into a
            # subdirectory added to PYTHONPATH. Two reasons: `python
            # script.py` already puts the script's own directory on
            # sys.path automatically, so no PYTHONPATH is needed for
            # imports to work; and some SPOC scripts locate their own
            # yaml config via a PYTHONPATH-derived path built without a
            # path separator, which silently degrades to the correct
            # bare filename when PYTHONPATH is unset/empty (relying on
            # cwd) but breaks if PYTHONPATH is set to anything else.
            if obs_builder_modules:
                for f in obs_builder_modules:
                    common.safe_copy(f, work_dir / f.name)

            # --- C. bufr input ---
            bufr_src = Path(bufr_path)
            if not bufr_src.is_file():
                logger.warning(f"  bufr file does not exist: {bufr_path}")
                group_failure = {"status": "missing_bufr_input"}
            else:
                local_bufr = common.safe_copy(bufr_src, work_dir / bufr_src.name)

                # --- D. run conversion ---
                split_var = repo_utils.find_split_variable(local_yaml)
                if split_var:
                    output_name = f"{script_prefix}_bq_ioda_{{splits/{split_var}}}.nc4"
                    output_glob = f"{script_prefix}_bq_ioda_*.nc4"
                    logger.info(f"  Config yaml declares split variable '{split_var}'; "
                                f"--output will include {{splits/{split_var}}}")
                else:
                    output_name = f"{script_prefix}_bq_ioda.nc4"
                    output_glob = None
                output_path = work_dir / output_name
                cmd = [
                    "python", str(local_script),
                    "--input", str(local_bufr),
                    "--output", str(output_path),
                ]

                logger.info("  $ " + " ".join(cmd))
                proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
                (work_dir / f"{script_prefix}_conversion.log").write_text(
                    f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
                )

                if output_glob is not None:
                    output_files = sorted(work_dir.glob(output_glob))
                else:
                    output_files = [output_path] if output_path.is_file() else []

                if proc.returncode != 0 or not output_files:
                    logger.error(f"  Conversion failed for {script_prefix} (exit {proc.returncode})")
                    group_failure = {"status": "conversion_failed", "conversion_returncode": proc.returncode}
                else:
                    logger.info(f"  OK -> {[p.name for p in output_files]}")

                    for f in output_files:
                        nc_utils.tag_with_spoc_provenance(f, ref, ref_type, matched_script.name)
                        if local_repo_status:
                            nc_utils.add_global_attrs(f, {"spoc_local_repo_git_status": local_repo_status})

        # --- F. distribute this group's output to each obs space ---
        # Assignment is computed once for the whole group (not per obs
        # space independently) so that ambiguous cases like 'sfc' vs
        # 'sfcship' both fuzzy-matching a label get resolved using the
        # full set of obs spaces this conversion produces, per Obs_dict.txt.
        output_assignment: Dict[str, Optional[Path]] = {}
        if group_failure is None:
            output_assignment = _assign_outputs_for_group(group_entries, output_files, script_prefix)

        for entry in group_entries:
            obs_space = entry.jedi_obs_space
            obs_dir = common.ensure_dir(test_dir / obs_space)
            (obs_dir / "SPOC_VERSION.txt").write_text(version_stamp)

            if group_failure is not None:
                results[obs_space] = {**group_failure, "conversion_group": script_prefix}
                continue

            spoc_match = output_assignment.get(obs_space)
            if spoc_match is None:
                available = [common.extract_split_label(f.name, script_prefix, SPOC_INFIX) for f in output_files]
                logger.warning(f"  [{obs_space}] could not be assigned a split output "
                                f"(available split labels for this group: {available}; "
                                f"other obs space(s) in this group may have claimed the "
                                f"best-matching one — see the group's other warnings)")
                results[obs_space] = {"status": "no_matching_split_output",
                                       "available_split_labels": available,
                                       "conversion_group": script_prefix}
                continue

            spoc_dest = obs_dir / f"{obs_space}_bq_ioda.nc4"
            common.safe_copy(spoc_match, spoc_dest, resolve_symlinks=False)
            nc_utils.add_global_attrs(spoc_dest, {"jedi_obs_space": obs_space})

            result: Dict[str, Any] = {
                "status": "ok",
                "subdir": str(obs_dir),
                "spoc_output": str(spoc_dest),
                "conversion_group": script_prefix,
                "repo_ref": ref,
                "repo_ref_type": ref_type,
            }

            results[obs_space] = result

    n_ok = sum(1 for r in results.values() if r.get("status") == "ok")
    logger.info(f"=== Stage 1 complete: {n_ok}/{len(entries)} obs spaces converted OK ===")
    return results
