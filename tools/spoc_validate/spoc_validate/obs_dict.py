"""Parsing for the user-maintained Obs_dict.txt config file.

Each non-blank, non-comment line has three whitespace-separated fields:

    <jedi_obs_space>  <spoc_script_basename>  <full_path_to_bufr_file>

e.g.:
    sondes prepbufr_adpupa.yaml /path/to/gdas1.20251129.12z.prepbufr
    pibal  prepbufr_adpupa.yaml /path/to/gdas1.20251129.12z.prepbufr

Multiple JEDI obs spaces can share the same script + bufr file — that's
expected when one SPOC conversion script splits its output across several
obs spaces (e.g. prepbufr_adpupa.py producing separate 'sonde' and 'pibal'
files from one prepbufr file). The second field's extension is ignored;
only its basename (prefix) is used to look up both the '<prefix>.py'
conversion script and the '<prefix>.yaml' config yaml in the SPOC repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ObsDictEntry:
    jedi_obs_space: str
    script_basename: str
    bufr_path: str

    @property
    def script_prefix(self) -> str:
        """script_basename with any extension stripped (e.g.
        'prepbufr_adpupa.yaml' or 'prepbufr_adpupa.py' -> 'prepbufr_adpupa'),
        used to look up both '<prefix>.py' and '<prefix>.yaml' in the SPOC
        repo regardless of which extension the user happened to write in
        Obs_dict.txt.
        """
        return Path(self.script_basename).stem


def load_obs_dict(path: str | Path) -> List[ObsDictEntry]:
    """Parse an Obs_dict.txt file into a list of ObsDictEntry."""
    path = Path(path)
    entries: List[ObsDictEntry] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"{path}:{lineno}: expected 3 whitespace-separated fields "
                f"(jedi_obs_space, script_basename, bufr_path), got "
                f"{len(parts)}: {line!r}"
            )
        entries.append(ObsDictEntry(*parts))

    seen = set()
    for e in entries:
        if e.jedi_obs_space in seen:
            raise ValueError(f"{path}: duplicate jedi_obs_space '{e.jedi_obs_space}'")
        seen.add(e.jedi_obs_space)

    return entries


def conversion_groups(entries: List[ObsDictEntry]) -> Dict[Tuple[str, str], List[ObsDictEntry]]:
    """Group entries that share the same conversion job (same script prefix
    + same bufr file), so the conversion only has to run once per unique
    pair even if several obs spaces reference it.

    Returns {(script_prefix, bufr_path): [entries...]}.
    """
    groups: Dict[Tuple[str, str], List[ObsDictEntry]] = {}
    for e in entries:
        key = (e.script_prefix, e.bufr_path)
        groups.setdefault(key, []).append(e)
    return groups
