"""Stage 3 — Set up hofx tests via SWELL (https://github.com/GEOS-ESM/swell).

Replaces the earlier manual background/ioda-staging approach: SWELL is
assumed to already be installed and on PATH (this module never tries to
fetch or install it). For each JEDI obs space in Obs_dict.txt, this creates
TWO swell hofx experiments — one to test against the ncdiag baseline
('<obs_space>_diag') and one to test the SPOC/bufr-query conversion
('<obs_space>_bq') — since swell itself handles background/ncdiag staging
once you run `swell launch`, this stage only needs to:

  1. Build the actual `-o` input for `swell create` by taking the
     user-provided experiment.template and substituting in the four
     experiment-specific values it can't know ahead of time
     (experiment_root, experiment_id, start_cycle_point,
     final_cycle_point — the cycle time parsed out of the obs space's bufr
     file path via regex) plus models.geos_atmosphere.observations (just
     the one obs space being tested), then run `swell create -o <filled-in
     template> hofx` — so swell processes the *entire* template through
     its own create/merge logic, not just a handful of override keys.
     Everything else, including `cycle_times` — so you can extend this to
     cycling/3dvar-style experiments beyond a single hofx cycle just by
     changing your template — comes from your experiment.template as-is;
     this stage deliberately doesn't touch it.
  2. As a belt-and-suspenders step, re-apply those same substitutions
     directly onto the experiment.yaml swell actually generated, in place
     — in case swell's own merge doesn't preserve them exactly as given.
  3. Write an activation script into the experiment directory for the user
     to run *after* `swell launch` — it copies the hofx test executable
     into the run directory, and (for _bq experiments, by default) swaps
     the observation file(s) for the SPOC/bufr-query converted IODA output.

Nothing here transfers backgrounds or ncdiags — `swell launch` does that.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from . import common
from . import obs_dict as obs_dict_mod

REQUIRED_KEYS = ["obs_dict_file", "test_directory", "experiment_template_file", "hofx_test_executable"]

SUFFIXES = ("bq", "diag")


# Shared, not-user-dependent spack-stack setup script — per bashrc_swell.txt
# this is the same for everyone, so it's a default rather than something
# every user has to supply. Still overridable via spack_stack_setup_script
# in config, in case it ever changes or differs for someone.
_DEFAULT_SPACK_STACK_SETUP_SCRIPT = (
    "/discover/nobackup/projects/gmao/advda/swell/jedi_modules/spackstack_1.9_intel_bundle"
)


def _build_swell_command(payload_cmd: "list[str]", config: Dict[str, Any]) -> "list[str]":
    """Wrap a swell command with this HPC's module-environment setup, if
    configured — 'module purge', source spack-stack, 'module use -a'/
    'module load' the swell module — all inside a single throwaway
    `bash -lc` subprocess (matching bashrc_swell.txt's own sequence,
    including the '-a' append flag on 'module use').

    No explicit restore/cleanup step is needed afterward, unlike your
    manual `module purge && source ~/.bashrc_swell` workflow: each
    subprocess.run() call here is already its own fresh process that
    inherits nothing from, and leaks nothing back into, this Python
    process's actual environment or any other subprocess call (including
    the plain `python <script>.py` conversion calls in grab_and_convert.py,
    which never touch swell's environment at all). The manual reset is
    only necessary because your interactive shell keeps that state around
    between commands in the same session — that problem doesn't exist here
    by construction.

    Controlled by two required-together config keys — set both to enable
    environment-switching, or leave both unset to run the command as-is
    (assumes you're already in the right environment):
      - swell_module_dir: directory passed to `module use -a`
      - swell_module_name: module name passed to `module load`
    spack_stack_setup_script is optional on top of that — defaults to the
    shared path above.
    """
    module_dir = config.get("swell_module_dir")
    module_name = config.get("swell_module_name")

    if not module_dir and not module_name:
        return payload_cmd
    if not (module_dir and module_name):
        raise KeyError(
            "Set both swell_module_dir and swell_module_name together to enable "
            f"swell environment-switching, or neither (got swell_module_dir="
            f"{module_dir!r}, swell_module_name={module_name!r})"
        )

    setup_script = config.get("spack_stack_setup_script", _DEFAULT_SPACK_STACK_SETUP_SCRIPT)

    payload = shlex.join(payload_cmd)
    bash_cmd = (
        f"module purge && "
        f"source {shlex.quote(setup_script)} && "
        f"module use -a {shlex.quote(module_dir)} && "
        f"module load {shlex.quote(module_name)} && "
        f"{payload}"
    )
    return ["bash", "-lc", bash_cmd]


def run_swell_create(override_input_path: Path, cwd: Path, logger, swell_cmd: str = "swell",
                      env_config: Optional[Dict[str, Any]] = None) -> subprocess.CompletedProcess:
    """Run `swell create -o <override_input_path> hofx`. override_input_path
    is expected to be a filled-in copy of the user's experiment.template
    (see build_filled_template) rather than a minimal hand-written file, so
    swell processes the whole template through its own create/merge logic.
    """
    payload_cmd = [swell_cmd, "create", "-o", str(override_input_path), "hofx"]
    cmd = _build_swell_command(payload_cmd, env_config or {})
    logger.info("  $ " + (cmd[-1] if cmd[0] == "bash" else " ".join(cmd)))
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _substitute_scalar_lines(lines: "list[str]", overrides: Dict[str, str], logger,
                              src_name: str) -> "list[str]":
    """Replace each `overrides` key's value in place wherever that key
    appears as a top-of-line 'key: value' scalar entry. Returns the
    modified line list; warns (doesn't raise) about any override key never
    found.
    """
    remaining = dict(overrides)
    out_lines = []
    for line in lines:
        matched_key = None
        for key in remaining:
            if re.match(rf"^\s*{re.escape(key)}:\s*.*$", line):
                matched_key = key
                break
        if matched_key:
            indent = re.match(r"^(\s*)", line).group(1)
            out_lines.append(f"{indent}{matched_key}: {remaining.pop(matched_key)}\n")
        else:
            out_lines.append(line)

    if remaining:
        logger.warning(f"  Scalar override key(s) not found as top-level lines in "
                        f"{src_name}, left unset: {list(remaining)}")
    return out_lines


def _replace_yaml_block_list(lines: "list[str]", key: str, new_items: "list[str]", logger,
                              src_name: str) -> "list[str]":
    """Find a 'key:' line that introduces a YAML block list (nothing after
    the colon but whitespace/a comment) and replace every '- item' line
    directly under it with new_items instead, preserving the list's
    original indentation. Used for models.geos_atmosphere.observations,
    which — unlike the 4 scalar overrides — is a nested list, not a
    same-line 'key: value' pair.

    Only replaces the first block-list occurrence of `key` found (this
    pipeline only supports a single model_component, so there's only one
    'observations:' block to begin with). Falls back to converting an
    inline 'key: [a, b]' or 'key: a' form to block style if found instead
    of a block list, so either input style still gets the right output.
    """
    key_block_re = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*(#.*)?$")
    key_inline_re = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*(?P<value>\S.*?)\s*(#.*)?$")

    out_lines: "list[str]" = []
    i = 0
    found = False
    while i < len(lines):
        line = lines[i]

        if not found:
            m = key_block_re.match(line)
            if m:
                found = True
                indent = m.group("indent")
                out_lines.append(line)
                i += 1
                item_indent = None
                while i < len(lines):
                    item = lines[i]
                    stripped = item.strip()
                    if not stripped:
                        break
                    cur_indent = len(item) - len(item.lstrip(" "))
                    if item_indent is None:
                        if stripped.startswith("-") and cur_indent >= len(indent):
                            item_indent = cur_indent
                        else:
                            break
                    if cur_indent == item_indent and stripped.startswith("-"):
                        i += 1  # drop the old item line
                        continue
                    break
                item_indent_str = " " * (item_indent if item_indent is not None else len(indent) + 2)
                for val in new_items:
                    out_lines.append(f"{item_indent_str}- {val}\n")
                continue

            m2 = key_inline_re.match(line)
            if m2:
                found = True
                indent = m2.group("indent")
                out_lines.append(f"{indent}{key}:\n")
                for val in new_items:
                    out_lines.append(f"{indent}  - {val}\n")
                i += 1
                continue

        out_lines.append(line)
        i += 1

    if not found:
        logger.warning(f"  List key '{key}:' not found in {src_name}; nothing replaced")
    return out_lines


def build_filled_template(template_path: Path, scalar_overrides: Dict[str, str],
                           list_overrides: Dict[str, "list[str]"], dest_path: Path, logger) -> None:
    """Copy the user's experiment.template to dest_path with the scalar
    experiment-specific overrides and the given YAML block-list overrides
    (e.g. models.geos_atmosphere.observations) substituted in — this
    becomes the actual `-o` input passed to `swell create`, so swell's own
    create/merge logic processes the whole template rather than just a
    handful of keys.
    """
    lines = Path(template_path).read_text().splitlines(keepends=True)
    lines = _substitute_scalar_lines(lines, scalar_overrides, logger, template_path.name)
    for key, items in list_overrides.items():
        lines = _replace_yaml_block_list(lines, key, items, logger, template_path.name)
    dest_path.write_text("".join(lines))


def patch_experiment_yaml(experiment_yaml: Path, scalar_overrides: Dict[str, str],
                           list_overrides: Dict[str, "list[str]"], logger) -> None:
    """Belt-and-suspenders: re-apply the same overrides (scalars + list
    overrides) directly onto the experiment.yaml swell actually generated,
    in place — in case swell's own override-merging doesn't preserve
    these particular values exactly as given (reformatting, dropping
    them, etc).
    """
    lines = experiment_yaml.read_text().splitlines(keepends=True)
    lines = _substitute_scalar_lines(lines, scalar_overrides, logger, experiment_yaml.name)
    for key, items in list_overrides.items():
        lines = _replace_yaml_block_list(lines, key, items, logger, experiment_yaml.name)
    experiment_yaml.write_text("".join(lines))


_ACTIVATION_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
"""Auto-generated by spoc_validate's hofx_swell stage for {experiment_id}.

Run this AFTER `swell launch` has completed for this experiment, to:
  1. copy the hofx test executable into the run directory
  2. (optional) replace the observation file(s) with the SPOC/bufr-query
     converted IODA output

Usage:
    python activate_hofx_run.py [--replace-obs | --no-replace-obs]
"""
import argparse
import shutil
import sys
from pathlib import Path

EXPERIMENT_ROOT = {experiment_root!r}
EXPERIMENT_ID = {experiment_id!r}
OBS_SPACE = {obs_space!r}
CYCLE_COMPACT = {cycle_compact!r}
HOFX_EXECUTABLE = {hofx_executable!r}
SPOC_IODA_FILE = {spoc_ioda_file!r}
DEFAULT_REPLACE_OBS = {default_replace_obs!r}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-obs", dest="replace_obs", action="store_true", default=None)
    parser.add_argument("--no-replace-obs", dest="replace_obs", action="store_false")
    args = parser.parse_args()
    replace_obs = DEFAULT_REPLACE_OBS if args.replace_obs is None else args.replace_obs

    run_dir = Path(EXPERIMENT_ROOT) / EXPERIMENT_ID / "run" / CYCLE_COMPACT / "geos_atmosphere"
    if not run_dir.is_dir():
        sys.exit(f"Run directory not found: {{run_dir}}\\n"
                 f"Has `swell launch` finished for this experiment yet?")

    exe_src = Path(HOFX_EXECUTABLE)
    if not exe_src.is_file():
        sys.exit(f"Configured hofx executable not found: {{exe_src}}")
    exe_dst = run_dir / exe_src.name
    shutil.copy2(exe_src, exe_dst)
    print(f"Copied hofx executable -> {{exe_dst}}")

    if replace_obs:
        if not SPOC_IODA_FILE:
            sys.exit("--replace-obs was requested but no SPOC IODA file is configured "
                      "for this experiment (this is expected for _diag experiments).")
        spoc_src = Path(SPOC_IODA_FILE)
        if not spoc_src.is_file():
            sys.exit(f"SPOC IODA file not found: {{spoc_src}}")
        matches = sorted(run_dir.glob(f"{{OBS_SPACE}}.*.nc4"))
        if not matches:
            sys.exit(f"No observation file matching '{{OBS_SPACE}}.*.nc4' found in {{run_dir}}")
        for obs_file in matches:
            backup = obs_file.with_name(obs_file.name + ".ncdiag_backup")
            if not backup.exists():
                shutil.copy2(obs_file, backup)
            shutil.copy2(spoc_src, obs_file)
            print(f"Replaced {{obs_file.name}} with SPOC/bufr-query output "
                  f"(original backed up to {{backup.name}})")
    else:
        print("Obs file replacement skipped (pass --replace-obs to swap in the "
              "SPOC/bufr-query file).")


if __name__ == "__main__":
    main()
'''


def write_activation_script(script_path: Path, *, experiment_root: str, experiment_id: str,
                             obs_space: str, cycle_compact: str, hofx_executable: str,
                             spoc_ioda_file: Optional[str], default_replace_obs: bool) -> None:
    content = _ACTIVATION_SCRIPT_TEMPLATE.format(
        experiment_id=experiment_id,
        experiment_root=experiment_root,
        obs_space=obs_space,
        cycle_compact=cycle_compact,
        hofx_executable=hofx_executable,
        spoc_ioda_file=spoc_ioda_file,
        default_replace_obs=default_replace_obs,
    )
    script_path.write_text(content)
    script_path.chmod(0o755)


def run(config: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    common.require_keys(config, REQUIRED_KEYS, context="hofx_swell config")

    test_dir = common.ensure_abs_dir(config["test_directory"])
    logger = common.get_logger("hofx_swell", test_dir / "hofx_swell.log")
    logger.info("=== Stage 3: Set up hofx tests via swell ===")

    entries = obs_dict_mod.load_obs_dict(config["obs_dict_file"])
    experiment_root = common.ensure_abs_dir(config.get("experiment_root", test_dir / "hofx_tests"))
    template_path = Path(config["experiment_template_file"])
    if not template_path.is_file():
        raise FileNotFoundError(f"experiment_template_file not found: {template_path}")
    hofx_executable = Path(config["hofx_test_executable"])
    swell_cmd = config.get("swell_command", "swell")
    overrides_dir = common.ensure_dir(test_dir / "_swell_overrides")

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for entry in entries:
        obs_space = entry.jedi_obs_space
        logger.info(f"--- {obs_space} ---")
        try:
            iso_time, cycle_compact = common.parse_cycle_from_bufr_path(entry.bufr_path)
        except ValueError as e:
            logger.warning(f"  {e}")
            results[obs_space] = {"default": {"status": "bad_cycle_time"}}
            continue

        spoc_output = test_dir / obs_space / f"{obs_space}_bq_ioda.nc4"
        spoc_output_str = str(spoc_output) if spoc_output.is_file() else None
        if spoc_output_str is None:
            logger.warning(f"  No SPOC output found at {spoc_output} "
                            f"(has Stage 1 been run for this obs space yet?) — the "
                            f"'{obs_space}_bq' activation script will have no SPOC file configured")

        obs_space_results: Dict[str, Any] = {}
        for suffix in SUFFIXES:
            experiment_id = f"{obs_space}_{suffix}"
            logger.info(f"  -- {experiment_id} --")

            overrides = {
                "experiment_root": str(experiment_root),
                "experiment_id": experiment_id,
                "start_cycle_point": f"'{iso_time}'",
                "final_cycle_point": f"'{iso_time}'",
            }
            list_overrides = {
                "observations": [obs_space],
            }

            filled_template_path = overrides_dir / f"{experiment_id}_override.yaml"
            build_filled_template(template_path, overrides, list_overrides, filled_template_path, logger)

            proc = run_swell_create(filled_template_path, cwd=test_dir, logger=logger, swell_cmd=swell_cmd,
                                     env_config=config)
            (overrides_dir / f"{experiment_id}_swell_create.log").write_text(
                f"CMD: {swell_cmd} create -o {filled_template_path} hofx\n\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
            )
            if proc.returncode != 0:
                logger.error(f"    swell create failed for {experiment_id} (exit {proc.returncode})")
                obs_space_results[suffix] = {"status": "swell_create_failed",
                                              "returncode": proc.returncode}
                continue

            experiment_dir = experiment_root / experiment_id
            suite_dir = experiment_dir / f"{experiment_id}-suite"
            experiment_yaml = suite_dir / "experiment.yaml"
            if not experiment_yaml.is_file():
                logger.error(f"    Expected {experiment_yaml} after swell create, but it "
                              f"doesn't exist — check {experiment_id}_swell_create.log; "
                              f"swell's directory-naming convention may differ from what "
                              f"this stage assumes (experiment_id as the folder name).")
                obs_space_results[suffix] = {"status": "missing_experiment_yaml",
                                              "expected_path": str(experiment_yaml)}
                continue

            patch_experiment_yaml(experiment_yaml, overrides, list_overrides, logger)
            logger.info(f"    Patched {experiment_yaml}")

            is_bq = suffix == "bq"
            script_path = experiment_dir / "activate_hofx_run.py"
            write_activation_script(
                script_path,
                experiment_root=str(experiment_root),
                experiment_id=experiment_id,
                obs_space=obs_space,
                cycle_compact=cycle_compact,
                hofx_executable=str(hofx_executable),
                spoc_ioda_file=spoc_output_str if is_bq else None,
                default_replace_obs=is_bq,
            )
            logger.info(f"    Wrote activation script -> {script_path}")

            obs_space_results[suffix] = {
                "status": "ok",
                "experiment_dir": str(experiment_dir),
                "suite_dir": str(suite_dir),
                "experiment_yaml": str(experiment_yaml),
                "activation_script": str(script_path),
                "cycle_time": iso_time,
            }

        results[obs_space] = obs_space_results

    n_ok = sum(1 for splits in results.values() for r in splits.values() if r.get("status") == "ok")
    n_total = sum(len(v) for v in results.values())
    logger.info(f"=== Stage 3 complete: {n_ok}/{n_total} experiments set up OK ===")
    return results
