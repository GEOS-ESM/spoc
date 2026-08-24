"""Helpers for fetching a SPOC tag and matching obs-type names to the
config yaml / conversion script / bufr / ncdiag files that belong to them.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

SPOC_REPO_URL = "https://github.com/GEOS-ESM/spoc.git"

# Directories inside the SPOC repo that hold what we need.
CONFIG_SUBDIR = "dump/config/atmosphere"
SCRIPTS_SUBDIR = "dump/scripts/atmosphere"


def _sanitize_ref(ref: str) -> str:
    """Make a git ref safe to use as a cache directory name (branches can
    contain slashes, e.g. 'feature/foo')."""
    return ref.replace("/", "__")


def get_local_repo_status(path: Path) -> Optional[str]:
    """Best-effort 'short HEAD (clean|dirty)' summary of a local SPOC
    checkout, for provenance purposes when using repo_local_path. Returns
    None (never raises) if git isn't available or the directory isn't a
    git repo — repo_local_path is still usable either way, this is just
    nice-to-have extra provenance detail.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{head} ({'dirty' if dirty else 'clean'})"
    except Exception:
        return None


def checkout_spoc_ref(ref: str, cache_dir: Path, ref_type: str = "tag",
                       logger: Optional[logging.Logger] = None,
                       repo_url: str = SPOC_REPO_URL) -> Path:
    """Sparse-checkout the given tag or branch of SPOC into
    cache_dir/<ref_type>/<ref>, fetching only the dump/config/atmosphere and
    dump/scripts/atmosphere directories. Returns the path to the checked-out
    repo.

    ref_type: "tag" or "branch".
      - "tag": tags are immutable, so an existing cached checkout is reused
        as-is.
      - "branch": branches move, so an existing cached checkout is updated
        (fetch + hard reset to origin/<branch>) rather than blindly reused,
        ensuring you always test the branch's current tip.
    """
    log = logger or logging.getLogger(__name__)
    ref_type = ref_type.lower()
    if ref_type not in ("tag", "branch"):
        raise ValueError(f"ref_type must be 'tag' or 'branch', got {ref_type!r}")

    dest = Path(cache_dir) / ref_type / _sanitize_ref(ref)

    def run(cmd: list[str]) -> None:
        log.debug("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=dest)

    if (dest / ".git").is_dir():
        if ref_type == "branch":
            log.info(f"Updating existing branch checkout '{ref}' at {dest} "
                      f"(branches move, so re-syncing to the current tip)")
            run(["git", "fetch", "origin", ref])
            run(["git", "reset", "--hard", f"origin/{ref}"])
        else:
            log.info(f"Reusing existing tag checkout '{ref}' at {dest} "
                     f"(tags are immutable, no re-fetch needed)")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning SPOC {ref_type} '{ref}' (sparse) into {dest}")

    run(["git", "clone", "--no-checkout", "--depth", "1",
         "--branch", ref, repo_url, "."])
    run(["git", "sparse-checkout", "init", "--cone"])
    run(["git", "sparse-checkout", "set", CONFIG_SUBDIR, SCRIPTS_SUBDIR])
    run(["git", "checkout", ref])

    return dest


def checkout_spoc_tag(repo_tag: str, cache_dir: Path,
                       logger: Optional[logging.Logger] = None,
                       repo_url: str = SPOC_REPO_URL) -> Path:
    """Backwards-compatible alias for checkout_spoc_ref(..., ref_type='tag')."""
    return checkout_spoc_ref(repo_tag, cache_dir, ref_type="tag",
                              logger=logger, repo_url=repo_url)


def _tokens(stem: str) -> list[str]:
    """Split a filename stem on underscores into lowercase tokens."""
    return [t.lower() for t in stem.split("_") if t]


def find_exact_file(basename: str, search_dir: Path, extension: str) -> Optional[Path]:
    """Find the file '<basename><extension>' in search_dir, where basename
    is expected to already be stripped of any extension (e.g. use
    ObsDictEntry.script_prefix). Used for Obs_dict-driven matching, where
    the user gives us the literal SPOC filename to look up rather than an
    obs-type token to fuzzy-match against.
    """
    candidate = Path(search_dir) / f"{basename}{extension}"
    return candidate if candidate.is_file() else None


def find_matching_file(obs_type: str, search_dir: Path, extension: str) -> Optional[Path]:
    """Find the single file in search_dir with the given extension whose
    filename (minus extension), split on underscores, contains obs_type as
    an exact token. This allows for an extra word attached before/after the
    match word (e.g. 'adpupa_prepbufr.yaml', 'raw_adpupa.yaml') while
    rejecting unrelated partial-substring matches (e.g. 'adpupabar.yaml').

    Returns None (and logs a warning) if there is no match or more than one
    candidate match.
    """
    log = logging.getLogger(__name__)
    obs_type_l = obs_type.lower()
    candidates = []
    for f in sorted(Path(search_dir).glob(f"*{extension}")):
        if obs_type_l in _tokens(f.stem):
            candidates.append(f)

    if not candidates:
        log.warning(f"No {extension} file matched obs type '{obs_type}' in {search_dir}")
        return None
    if len(candidates) > 1:
        log.warning(
            f"Multiple {extension} files matched obs type '{obs_type}' in "
            f"{search_dir}: {[c.name for c in candidates]}. Skipping ambiguous match."
        )
        return None

    return candidates[0]


def find_split_variable(yaml_path: Path) -> Optional[str]:
    """Look for a top-level 'splits:' line in a SPOC config yaml and return
    the variable name declared immediately below it (e.g. 'obsType'),
    whether it's written as a dict key ('obsType:') or a list item
    ('- obsType'). Returns None if the yaml has no splits section.

    Done as a light text scan rather than a full structural yaml parse,
    since the spec is just "right below the splits: line" and split
    section formatting varies across SPOC config files.
    """
    lines = Path(yaml_path).read_text().splitlines()
    for i, line in enumerate(lines):
        if re.match(r'^\s*splits\s*:\s*(#.*)?$', line):
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.strip().startswith("#"):
                    continue
                token = nxt.strip().lstrip("-").strip()
                token = re.split(r'[:\s]', token, maxsplit=1)[0]
                return token or None
            return None
    return None


def find_obs_builder_modules(scripts_dir: Path) -> list[Path]:
    """Find every script in scripts_dir whose name ends in 'obs_builder.py'
    (i.e. 'obs_builder' appears immediately before the .py extension, e.g.
    'bufr_obs_builder.py', 'adpupa_obs_builder.py', 'obs_builder.py'). These
    are shared helper modules some SPOC conversion scripts import.
    """
    return sorted(Path(scripts_dir).glob("*obs_builder.py"))
