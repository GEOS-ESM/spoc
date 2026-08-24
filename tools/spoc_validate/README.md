# spoc_validate

Quick validation pipeline for new tags/branches of
[SPOC](https://github.com/GEOS-ESM/spoc) (the bufr->IODA observation
pre-processor). Runs on a Unix HPC login/compute node. Assumes
[swell](https://github.com/GEOS-ESM/swell) is already installed and on
`PATH` for Stage 3 — nothing here fetches or installs swell itself.

## Layout

```
spoc_validate/
├── spoc_validate/           # library code
│   ├── common.py            # config loading, logging, safe copy, fuzzy label matching
│   ├── obs_dict.py          # Obs_dict.txt parsing
│   ├── repo_utils.py        # git ref checkout + exact/fuzzy file matching
│   ├── nc_utils.py          # netCDF provenance tagging + QC-based inventory
│   ├── grab_and_convert.py  # Stage 1
│   ├── compare_files.py     # Stage 2 (optional)
│   └── hofx_swell.py        # Stage 3 (swell-based)
├── scripts/
│   ├── 01_grab_and_convert.py
│   ├── 02_compare_files.py
│   └── 03_setup_hofx_swell.py
├── config/
│   ├── grab_convert_config.yaml   # used by scripts 01 and 02
│   ├── hofx_swell_config.yaml     # used by script 03
│   └── Obs_dict.txt               # copy of your real obs-space mapping
├── notebooks/
│   └── 04_analyze_hofx_output.ipynb   # Stage 4, run in JupyterLab — SEE CAVEAT BELOW
└── requirements.txt
```

All three files under `config/` are pre-populated with your actual current
settings (paths, `repo_local_path`, `swell_module_dir`/`swell_module_name`,
`experiment_template_file`, and so on) rather than generic placeholders —
each time this pipeline changes, your real values carry forward and only
genuinely new/renamed options get added, so you shouldn't need to
re-enter anything that already worked. `obs_dict_file` in both config
files points at `/discover/nobackup/jemccurr/spoc_validate/config/Obs_dict.txt`
— i.e. wherever you extract this package on Discover, `config/Obs_dict.txt`
*is* the file both stages read, not a separate copy — so editing it there
(to add/remove obs spaces) takes effect directly with no path to keep in
sync.

## The Obs_dict.txt file

Everything is now driven by a single user-maintained file — plain text,
one line per JEDI obs space, three whitespace-separated fields:

```
<jedi_obs_space> <spoc_script_basename> <full_path_to_bufr_file>
```

e.g.:

```
sondes prepbufr_adpupa.yaml /path/to/gdas1.20251129.12z.prepbufr
pibal  prepbufr_adpupa.yaml /path/to/gdas1.20251129.12z.prepbufr
sfc    prepbufr_sfc.yaml    /path/to/gdas1.20251129.12z.prepbufr
```

The second field's extension is ignored — only its prefix
(`prepbufr_adpupa`) is used to look up both `<prefix>.py` (conversion
script) and `<prefix>.yaml` (config) in the SPOC repo's
`dump/scripts/atmosphere` and `dump/config/atmosphere`, by **exact
filename**, not fuzzy matching. Multiple obs spaces sharing the same
script + bufr file is the normal case for a script that splits its output
(e.g. `prepbufr_adpupa.py` -> separate `sonde` and `pibal` files) — Stage 1
only runs that conversion once per unique (script, bufr file) pair and
distributes the resulting split file(s) out to whichever obs space(s)
reference them.

## Stages

**1. Grab Tag and Convert** (`01_grab_and_convert.py`) — for each unique
(script, bufr file) pair in `obs_dict_file`: sparse-checkouts the given
SPOC ref (`repo_tag`, `repo_branch`, or `repo_local_path` for an
uncommitted local checkout), grabs the exact-named config yaml +
conversion script, copies any shared `*obs_builder.py` helper modules into
the same working directory (no `PYTHONPATH` — see notes below), runs the
conversion (handling `{splits/...}` output automatically where the yaml
declares a split variable), and tags output with SPOC provenance. Each
JEDI obs space then gets its own `test_directory/<obs_space>/` directory
containing just `<obs_space>_bq_ioda.nc4` — the split resolved to a
single obs space's file, using fuzzy matching between the obs space name
and the actual split labels produced (see notes below).

```bash
python scripts/01_grab_and_convert.py --config config/grab_convert_config.yaml
```

**2. Compare Files** (optional, `02_compare_files.py`) — for each obs
space in `obs_dict_file`, locates its ncdiag comparison file via
`experiment_template_file`: reads `models.<model>.ioda_locations_not_in_r2d2`
out of that swell experiment template, then looks for
`<that path>/<cycle_compact>/geos_atmosphere/<obs_space>.<file_time>.nc4`
— the directory named for the true cycle time (parsed from the obs
space's bufr filename, same as Stage 3), but the filename's own
timestamp 3 hours behind that (`file_time` = `cycle_compact` − 3h, e.g.
directory `20251129T120000Z` but filename `sondes.20251129T090000Z.nc4`)
— and stages it as `<obs_space>_ncdiag_ioda.nc4`: exact, per-obs-space,
no manually pre-populated ncdiag directory or substring matching needed.
Compares that against `<obs_space>_bq_ioda.nc4` from Stage 1, scoped to
the `ObsValue` group and its paired quality-marker group (`QualityMarker`
/ `PreQC`, auto-detected). Writes a `<obs_space>_qc_inventory.csv` +
`<obs_space>_discrepancies.txt` per obs space.

```bash
python scripts/02_compare_files.py --config config/grab_convert_config.yaml
```

**3. Set up hofx tests via swell** (`03_setup_hofx_swell.py`) — for each
JEDI obs space, creates **two** swell hofx experiments:
`<obs_space>_bq` (to test the SPOC/bufr-query conversion) and
`<obs_space>_diag` (the ncdiag baseline). For each:

1. Builds the actual `-o` input for `swell create` by taking your
   `experiment_template_file` and substituting in `experiment_root`,
   `experiment_id`, `start_cycle_point`/`final_cycle_point` (the cycle
   time parsed out of the obs space's bufr filename via regex, handling
   both `YYYYMMDD` and the real operational `YYMMDD` date formats, and
   both `HHz`/`tHHz` hour forms), plus rewriting
   `models.geos_atmosphere.observations` to just the one obs space being
   tested (a nested YAML list, not a same-line `key: value` pair, so
   handled separately from the four scalar overrides — see notes below)
   — then runs `swell create -o <filled-in template> hofx`, so swell
   processes your *entire* template through its own create/merge logic
   rather than just a handful of override keys. Everything else in your
   template — including `cycle_times`, so you can extend this to
   cycling/3DVar-style experiments just by changing your template — is
   left alone.
2. As a belt-and-suspenders step, re-applies those same substitutions
   directly onto the `experiment.yaml` swell actually generated, in
   place — in case swell's own merge doesn't preserve them exactly as
   given.
3. Writes `activate_hofx_run.py` into `hofx_tests/<experiment_id>/` for
   you to run **after** `swell launch` finishes for that experiment. It
   copies the configured `hofx_test_executable` into
   `<experiment_root>/<experiment_id>/run/<cycle>/geos_atmosphere/`, and —
   for `_bq` experiments by default — replaces any observation file(s)
   matching `<obs_space>.*.nc4` in that directory with the SPOC/bufr-query
   IODA output, backing up the original first (`--no-replace-obs` to skip
   this; `--replace-obs`/`--no-replace-obs` also work as explicit flags on
   either script if you want to override the default).

Nothing here transfers backgrounds or ncdiags — `swell launch` handles
that itself, for both `_bq` and `_diag` experiments identically (the `_bq`
activation script's job is just to swap the observation file afterward).

```bash
python scripts/03_setup_hofx_swell.py --config config/hofx_swell_config.yaml
# ... for each experiment printed in the summary:
swell launch <experiment_root>/<experiment_id>/<experiment_id>-suite
python <experiment_root>/<experiment_id>/activate_hofx_run.py
```

**4. Analyze Swell hofx output** (`notebooks/04_analyze_hofx_output.ipynb`)
— ⚠️ **not yet updated for this rework.** It still assumes the old
`test_directory/<obs_type>/<run_name>[_<label>]/` directory layout from
before Stage 3 moved to swell. The new hofx run output lives under
`<experiment_root>/<experiment_id>/run/<cycle>/geos_atmosphere/` instead,
and the actual output filename/format from the swell-launched hofx
executable hasn't been confirmed yet. This needs a follow-up pass once
you've actually run a swell hofx experiment and can show me what comes out
— flagging this now rather than guessing at a notebook rewrite blind.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use your HPC's module system
```

`git` must be able to reach GitHub from wherever you run Stage 1 (unless
using `repo_local_path`). `swell`
must be installed and on `PATH` (or point `swell_command` in
`hofx_swell_config.yaml` at it) for Stage 3.

If your HPC's module environment for `swell` conflicts with the one used
for the SPOC/bufr-query conversion (Stages 1/2), set `swell_module_dir` +
`swell_module_name` in `hofx_swell_config.yaml` (see the comments there —
`spack_stack_setup_script` has a shared default and usually doesn't need
setting). No cleanup step is needed afterward; see the note under "Notes"
below for why.

## Notes / assumptions worth double-checking against your setup

- **`observations:` list override**: unlike the four scalar overrides
  (`experiment_root`, `experiment_id`, `start_cycle_point`,
  `final_cycle_point` — all same-line `key: value` pairs),
  `models.geos_atmosphere.observations` is a nested YAML block list, so it
  needs different handling: the `observations:` line is found, every
  existing `- item` line under it is dropped, and a single
  `- <obs_space>` line is written in its place, at whatever indentation
  the list actually used (same-level-as-key and more-indented-than-key
  styles both handled — this file uses same-level, e.g.
  `observations:` / `- sfc` at matching indentation, which is also how
  `model_components:` and `cycle_times:` are written here — those two are
  left untouched, along with everything else in your template; only
  `observations:` gets rewritten). An inline `observations: [sfc]` or
  `observations: sfc` form gets converted to block style rather than left
  inline. Verified against your real `experiment.yaml`, output re-parsed
  with PyYAML to confirm validity for `sondes`, `sfcship`, and
  `aircraft_temperature`. **Known limitation**: this only handles a
  single model component — if `model_components` ever lists more than
  one model, only the first `observations:` block encountered gets
  rewritten, since there'd be no way to tell which block belongs to
  which model from a plain line scan without deeper YAML-aware parsing.
  Not an issue for the current one-model (`geos_atmosphere`) setup.
- **`experiment_id`/`-suite` naming**: confirmed — `experiment_id`
  (including the `_bq`/`_diag` suffix) is the directory name
  (`hofx_tests/sondes_bq/`), and swell names the suite directory after
  `experiment_id` too (`sondes_bq-suite`). No longer a guess.
- **ncdiag filename's -3h offset** (Stage 2, `common.offset_compact_time`):
  the ncdiag directory under `ioda_locations_not_in_r2d2` is named for the
  true cycle time, but the file *inside* it is timestamped 3 hours
  earlier — e.g. cycle `2025-11-29T12:00:00Z` means directory
  `20251129T120000Z/geos_atmosphere/` but filename
  `sondes.20251129T090000Z.nc4`. Verified against a cross-midnight case
  too (`...T010000Z` − 3h → previous day `...T220000Z`) to make sure the
  date rolls back correctly, not just the hour.
- **Cycle time extraction** (`common.parse_cycle_from_bufr_path`, shared
  between Stage 2's ncdiag lookup and Stage 3's cycle-point overrides):
  regex `\.(\d{8}|\d{6})\.t?(\d{2})z` pulled from the bufr file path,
  handling both `YYYYMMDD` and the actual operational `YYMMDD` naming
  convention (two-digit years expanded as `20YY`) — tested against
  `gdas1.20251129.12z.prepbufr`, `gdas1.20251129.t12z.prepbufr.acft_profiles`,
  `gdas1.251129.12z...`, and `gdas1.251129.t12z...`, all resolving to
  `2025-11-29T12:00:00Z`. `start_cycle_point` and `final_cycle_point` are
  set to the same value (a single-cycle hofx test). If a bufr filename
  doesn't match this pattern at all, that obs space is skipped with a
  warning rather than guessing.
- **Obs-space-to-split-label matching** (Stage 1) is done at the
  conversion-group level, not per obs space independently: every obs
  space Obs_dict.txt maps to a given (script, bufr file) pair is matched
  against that group's actual split output labels all at once, using
  best-score-first greedy assignment (exact match → simple pluralization
  → substring containment, longer match preferred as a tie-breaker), with
  each label claimed by at most one obs space. This resolves cases that
  would be ambiguous if matched independently — e.g. `sfc` and `sfcship`
  both substring-match a hypothetical `sfc`/`ship` label pair, but
  group-level assignment lets `sfc` claim its exact match first, leaving
  `sfcship` to claim `ship`. Confirmed against your real `Obs_dict.txt`
  (`sondes`/`pibal` from `prepbufr_adpupa`, `sfc`/`sfcship` from
  `prepbufr_sfc`, `aircraft_wind`/`aircraft_temperature` from
  `prepbufr_aircraft`) with fabricated split labels standing in for the
  real ones. If an obs space still can't be assigned (no candidate label
  scores above the "no match" threshold), it's skipped with a warning
  rather than guessed — check the Stage 1 log for these.
- **Stage 1 cleans up before each run**: everything Stage 1 owns under
  `test_directory` (every `<obs_space>/` directory, `_conversions/`,
  `SPOC_VERSION.txt`, its own log) is removed at the start of every run,
  before anything new is written — so re-running against the same
  `test_directory` can't leave stale data (an obs space removed from
  Obs_dict.txt, leftover split files from a since-changed conversion
  script, etc.) around to confuse later stages. `hofx_tests/`,
  `_swell_overrides/` (Stage 3's) and `.spoc_repo_cache/` (meant to
  persist) are left untouched.
- **Repo checkout caching**: unchanged from before — tag checkouts are
  reused as-is; branch checkouts are `git fetch` + hard-reset to
  `origin/<branch>` on every run.
- **`repo_local_path`**: for testing experimental changes in a local SPOC
  checkout before pushing anywhere. No git operations happen at all beyond
  a best-effort `git rev-parse --short HEAD` + `git status --porcelain`
  read (never required to succeed) for provenance — the directory's
  current on-disk state is used exactly as-is, uncommitted changes
  included. Recorded in `SPOC_VERSION.txt` and as a
  `spoc_local_repo_git_status` netCDF attribute (e.g. `"e40dfe9 (dirty)"`)
  on every output file when available.
- **Swell module environment**: set `swell_module_dir` +
  `swell_module_name` together (in `hofx_swell_config.yaml`) to run
  `swell create` inside its own `bash -lc "module purge && source
  <spack-stack> && module use -a <dir> && module load <name> && swell
  create ..."` subprocess instead of running the bare command — matching
  your `bashrc_swell.txt` sequence exactly, `-a` flag included.
  `spack_stack_setup_script` defaults to the shared project path from
  `bashrc_swell.txt` (same for everyone) and only needs overriding if
  yours differs. Deliberately **not** implemented as "purge + source
  `~/.bashrc_swell` before, purge + source `~/.bashrc` after" the way the
  manual workflow needs: that reset is only necessary because an
  interactive shell keeps sourced state around for every later command in
  the same session. Every subprocess call this pipeline makes — the
  wrapped `swell create` and the plain `python <script>.py` conversion
  calls in Stage 1 alike — is already its own fresh process, inheriting
  only from this Python process's real environment (which nothing here
  ever mutates) and not from any sibling subprocess call. So the
  environments never actually collide in the first place, and there's
  nothing to explicitly clean up afterward.
  operational `YYMMDD` bufr filename convention (e.g. `gdas1.251129.12z...`
  for 2025-11-29), expanding two-digit years as `20YY` — safe for these
  files since none predate 2000.
- **Fill-value detection / QC group auto-detection / total obs count**
  (Stage 2): unchanged from before this rework — see `nc_utils.py` if you
  need the details; `QualityMarker` (SPOC) / `PreQC` (ncdiag) are
  auto-detected per file. What changed is *how the ncdiag file itself
  gets found* — see the Stage 2 description above — not this part.
- **Shared helper modules**: any script in `dump/scripts/atmosphere` whose
  name ends in `obs_builder.py` is copied flat into the same working
  directory as the conversion script and its yaml (not a subdirectory, and
  not via `PYTHONPATH`). `python script.py` already puts the script's own
  directory on `sys.path`, so no `PYTHONPATH` is needed for the import to
  work — and setting `PYTHONPATH` actively broke some conversion scripts
  early on: at least one locates its own yaml config via a path built by
  concatenating `PYTHONPATH`'s first entry directly onto the yaml filename
  with no separator, which silently degrades to just the bare filename
  (resolved against `cwd`, which is already the working directory) when
  `PYTHONPATH` is unset, but breaks the moment `PYTHONPATH` is set to
  anything else. Keeping everything flat in one directory with no
  `PYTHONPATH` override sidesteps this regardless of which scripts do or
  don't have that quirk.
- **`SPOC_VERSION.txt`**: written into `test_directory/` and into every
  `<obs_space>/` subdirectory, recording the SPOC `repo_ref`/`ref_type`
  and the UTC time of the grab/convert run.
- **Absolute paths internally**: `test_directory` is resolved to an
  absolute path as soon as Stage 1 or Stage 3 starts, even if given as a
  relative path in the config — avoids relative-path/subprocess-cwd bugs
  (see git history if curious; this bit us once already with `PYTHONPATH`).
- **Split output handling** (Stage 1): unchanged mechanically from before
  this rework — a config yaml's `splits:` section is detected and the
  `--output` argument gets a `{splits/<variable>}` placeholder
  automatically; whatever file(s) actually come out get discovered by
  globbing rather than assuming a literal filename.
