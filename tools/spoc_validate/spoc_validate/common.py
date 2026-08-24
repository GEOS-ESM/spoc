"""Shared helpers used across all pipeline stages."""

from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} did not parse to a dict")
    return cfg


def get_logger(name: str, log_file: str | Path | None = None,
                level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout and, optionally, a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def safe_copy(src: str | Path, dst: str | Path, resolve_symlinks: bool = True) -> Path:
    """Copy src to dst, resolving symlinks (soft links used for bufr/ncdiag
    staging directories) so the actual file content is copied rather than a
    dangling link relative to a different directory.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    real_src = src.resolve() if resolve_symlinks else src
    if not real_src.is_file():
        raise FileNotFoundError(f"Source file does not exist (resolved: {real_src})")

    shutil.copy2(real_src, dst)
    return dst


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_abs_dir(path: str | Path) -> Path:
    """Like ensure_dir, but resolves to an absolute path first. Use this for
    any directory whose path might later be embedded in a subprocess's
    PYTHONPATH/cwd or written into a config file — a relative path there
    gets silently re-interpreted against whatever the subprocess's cwd
    happens to be, which is a common source of hard-to-diagnose 'module not
    found' errors.
    """
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_split_label(filename: str | Path, obs_type: str, infix: str) -> "str | None":
    """Extract the split label from a filename following the
    '<obs_type><infix><label>.nc4' naming convention (infix='_bq_ioda_' for
    SPOC conversion output, infix='_ncdiag_ioda_' for split ncdiag input,
    if you ever have per-split ncdiag files). Returns None for the plain,
    non-split naming (e.g. '<obs_type>_bq_ioda.nc4').
    """
    stem = Path(filename).stem
    prefix = f"{obs_type}{infix}"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return None


def match_split_label(obs_space: str, label: str) -> bool:
    """Loose match between a JEDI obs space name (from Obs_dict.txt) and a
    split label a SPOC conversion script actually produced, tolerating
    minor naming differences: exact match, simple pluralization ('sondes'
    vs 'sonde'), or substring containment either direction ('aircraft_wind'
    vs 'wind').
    """
    a, b = obs_space.lower(), label.lower()
    if a == b:
        return True
    a_variants = {a, a.rstrip("s")}
    b_variants = {b, b.rstrip("s")}
    if a_variants & b_variants:
        return True
    return a in b or b in a


def _label_match_score(obs_space: str, label: str) -> "int | None":
    a, b = obs_space.lower(), label.lower()
    if a == b:
        return 100
    a_variants = {a, a.rstrip("s")}
    b_variants = {b, b.rstrip("s")}
    if a_variants & b_variants:
        return 90
    if a in b or b in a:
        return 50 + len(label)
    return None


def assign_labels(obs_spaces: "list[str]", labels: "list[str]") -> "dict[str, str | None]":
    """Assign each obs_space in a conversion group to at most one label
    (and vice versa), using every obs_space/label in the group at once
    rather than matching each obs_space independently.

    Matching independently breaks down when a label fuzzy-matches more
    than one obs_space in the same group — e.g. obs spaces 'sfc' and
    'sfcship' both substring-match a hypothetical 'sfc' label. Considering
    the whole group at once resolves this the way you'd expect: candidate
    (obs_space, label) pairs are scored (exact match highest, then simple
    pluralization, then substring containment with longer labels
    preferred as a tie-breaker) and assigned greedily best-score-first,
    removing both the obs_space and the label from further consideration
    once assigned — so 'sfc' claims its exact match first, leaving
    'sfcship' to claim whatever's left over instead of both claiming the
    same label.

    Returns {obs_space: label_or_None}; a None value means no label could
    be confidently assigned to that obs_space.
    """
    candidates = []
    for obs_space in obs_spaces:
        for label in labels:
            score = _label_match_score(obs_space, label)
            if score is not None:
                candidates.append((score, obs_space, label))
    candidates.sort(key=lambda c: -c[0])

    assigned: "dict[str, str]" = {}
    used_labels = set()
    for score, obs_space, label in candidates:
        if obs_space in assigned or label in used_labels:
            continue
        assigned[obs_space] = label
        used_labels.add(label)

    return {obs_space: assigned.get(obs_space) for obs_space in obs_spaces}


def require_keys(cfg: Dict[str, Any], keys: list[str], context: str = "config") -> None:
    """Raise a clear error if required keys are missing from a config dict."""
    missing = [k for k in keys if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise KeyError(f"Missing required key(s) in {context}: {missing}")


# Matches both 'gdas1.20251129.12z.prepbufr' (YYYYMMDD) and
# 'gdas1.251129.t12z.prepbufr.acft_profiles' (YYMMDD — the actual
# operational naming convention) styles, and both 'HHz'/'tHHz' hour forms.
# Shared between hofx_swell.py (cycle_point overrides) and compare_files.py
# (locating ncdiag files by cycle).
_CYCLE_RE = re.compile(r"\.(\d{8}|\d{6})\.t?(\d{2})z", re.IGNORECASE)


def parse_cycle_from_bufr_path(bufr_path: str) -> "tuple[str, str]":
    """Extract the cycle date/hour from a bufr file path via regex.
    Handles both 8-digit (YYYYMMDD) and 6-digit (YYMMDD, expanded to
    20YYMMDD — always valid for these operational files, which don't span
    pre-2000 dates) date formats. Returns (iso_time, compact_time), e.g.
    ('2025-11-29T12:00:00Z', '20251129T120000Z').
    """
    m = _CYCLE_RE.search(str(bufr_path))
    if not m:
        raise ValueError(f"Could not extract a cycle date/time from bufr path: {bufr_path}")
    date_str, hour_str = m.groups()
    if len(date_str) == 6:
        year, month, day = f"20{date_str[0:2]}", date_str[2:4], date_str[4:6]
    else:
        year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
    iso_time = f"{year}-{month}-{day}T{hour_str}:00:00Z"
    compact_time = f"{year}{month}{day}T{hour_str}0000Z"
    return iso_time, compact_time


def offset_compact_time(compact_time: str, hours: int) -> str:
    """Shift a 'YYYYMMDDTHHMMSSZ' compact time by the given number of
    hours (may be negative), returning the same format. Used for the
    ncdiag filename convention, where the file's own timestamp is 3 hours
    behind the actual cycle time the directory it lives in is named for.
    """
    dt_obj = dt.datetime.strptime(compact_time, "%Y%m%dT%H%M%SZ")
    dt_obj += dt.timedelta(hours=hours)
    return dt_obj.strftime("%Y%m%dT%H%M%SZ")
