"""netCDF4 helpers shared by the grab/convert and compare stages."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import netCDF4 as nc
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "netCDF4 is required (pip install netCDF4 / module load netcdf-python "
        "on your HPC system)."
    ) from e

# Sentinel values commonly used as "fill"/missing in IODA & GSI-diag netCDFs,
# checked in addition to any explicit _FillValue attribute.
_KNOWN_SENTINELS = (
    9.969209968386869e36,   # netCDF4 default float fill
    -9999.0,
    -999.0,
    1.0e15,
    1.0e10,
    3.4028235e38,           # netCDF4 default float32 fill
    -2147483647,            # netCDF4 default int fill
)
_SENTINEL_ABS_THRESHOLD = 1.0e9  # treat |value| beyond this as "extreme"


def add_global_attrs(nc_path: str | Path, attrs: Dict[str, Any]) -> None:
    """Append global attributes (e.g. provenance: SPOC tag, conversion
    timestamp) to an existing netCDF file, opened in append mode.
    """
    with nc.Dataset(nc_path, mode="a") as ds:
        for key, value in attrs.items():
            ds.setncattr(key, value)


def tag_with_spoc_provenance(nc_path: str | Path, ref: str, ref_type: str,
                              conversion_script: str) -> None:
    """Convenience wrapper: stamp the output file with the SPOC tag/branch
    used to produce it, the conversion script name, and a UTC timestamp.

    ref_type is "tag" or "branch". spoc_repo_tag is kept (set to ref) for
    backwards compatibility with anything reading that attribute from
    files produced before branch support was added.
    """
    add_global_attrs(nc_path, {
        "spoc_repo_ref": ref,
        "spoc_repo_ref_type": ref_type,
        "spoc_repo_tag": ref,  # legacy attr name, kept for compatibility
        "spoc_conversion_script": conversion_script,
        "spoc_validate_conversion_time_utc": dt.datetime.utcnow().isoformat() + "Z",
    })


def _iter_variables(group: "nc.Group", prefix: str = "") -> Iterable[tuple[str, "nc.Variable"]]:
    """Recursively walk an IODA-style grouped netCDF4 Dataset, yielding
    (full_path, variable) pairs, e.g. ('ObsValue/airTemperature', <var>).
    """
    for name, var in group.variables.items():
        yield f"{prefix}{name}", var
    for gname, subgroup in group.groups.items():
        yield from _iter_variables(subgroup, prefix=f"{prefix}{gname}/")


def _detect_fill_value(var: "nc.Variable", data: np.ndarray) -> Any:
    """Determine the fill/missing value for a variable: prefer an explicit
    _FillValue/missing_value attribute; otherwise fall back to checking for
    known sentinel values that appear in the data.
    """
    for attr_name in ("_FillValue", "missing_value"):
        if attr_name in var.ncattrs():
            return var.getncattr(attr_name)

    if not np.issubdtype(data.dtype, np.number):
        return None

    flat = data.ravel()
    flat = flat[~np.isnan(flat.astype("float64", copy=False))] if np.issubdtype(
        data.dtype, np.floating) else flat
    if flat.size == 0:
        return None

    # Check known sentinels first (fast path).
    for sentinel in _KNOWN_SENTINELS:
        if np.any(np.isclose(flat, sentinel, rtol=1e-6)):
            return sentinel

    # Fall back to: the most common value, IF it's "extreme" (large magnitude)
    # and repeats often enough to plausibly be a fill value rather than a
    # legitimate physical value.
    values, counts = np.unique(flat, return_counts=True)
    if values.size == 0:
        return None
    most_common_idx = np.argmax(counts)
    candidate = values[most_common_idx]
    candidate_count = counts[most_common_idx]
    if abs(candidate) > _SENTINEL_ABS_THRESHOLD and candidate_count > 1:
        return candidate

    return None


def summarize_variables(nc_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Walk every variable in an IODA netCDF file and return a dict keyed by
    full group/variable path with dtype, shape, detected fill value, and the
    mean of the data excluding NaNs and the detected fill value.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    with nc.Dataset(nc_path, mode="r") as ds:
        for path, var in _iter_variables(ds):
            try:
                data = var[:]
                data = np.ma.filled(data, np.nan) if np.ma.isMaskedArray(data) else data
                data = np.asarray(data)
            except Exception as e:  # pragma: no cover - defensive
                summary[path] = {"error": f"could not read variable: {e}"}
                continue

            fill_value = _detect_fill_value(var, data)

            mean_val = None
            if np.issubdtype(data.dtype, np.number):
                work = data.astype("float64", copy=True).ravel()
                mask = ~np.isnan(work)
                if fill_value is not None:
                    mask &= ~np.isclose(work, float(fill_value), rtol=1e-6)
                valid = work[mask]
                if valid.size > 0:
                    mean_val = float(np.mean(valid))

            summary[path] = {
                "dtype": str(var.dtype),
                "shape": tuple(var.shape),
                "fill_value": fill_value,
                "mean_excl_fill": mean_val,
            }
    return summary


# --- QC-aware inventory (ObsValue + paired quality-marker group) ---

OBSVALUE_GROUP = "ObsValue"

# Different IODA producers spell the quality-marker group differently.
# Checked in this order; the first one present in the file wins.
QC_GROUP_CANDIDATES = ["QualityMarker", "Quality_Marker", "Quality_marker", "PreQC", "QCMarker"]

# Dimension name IODA v2+ uses for "number of observations". If it's not
# present, total obs count falls back to the length of the first ObsValue
# variable found.
OBS_COUNT_DIMENSION = "Location"


def _valid_mask(data: np.ndarray, fill_value: Any) -> np.ndarray:
    """Boolean mask of entries that are neither NaN nor the detected fill
    value."""
    work = data.astype("float64", copy=False)
    mask = ~np.isnan(work)
    if fill_value is not None:
        mask &= ~np.isclose(work, float(fill_value), rtol=1e-6)
    return mask


def _total_obs_count(ds: "nc.Dataset", obsvalue_group: "nc.Group") -> Optional[int]:
    if OBS_COUNT_DIMENSION in ds.dimensions:
        return int(ds.dimensions[OBS_COUNT_DIMENSION].size)
    for var in obsvalue_group.variables.values():
        return int(var.shape[0])
    return None


def inventory_obs_file(nc_path: str | Path) -> Dict[str, Any]:
    """Build a QC-aware inventory of an IODA file's ObsValue group:

      - total_obs: total number of observations in the file (from the
        'Location' dimension, or the first ObsValue variable's length if
        that dimension isn't present)
      - qc_group_used: whichever QC_GROUP_CANDIDATES name was found in the
        file (None if none of them are present)
      - variables: per-ObsValue-variable dict with:
          n_qc_le_2         - count with quality marker <= 2
          n_qc_gt_2         - count with quality marker > 2
          mean_value_qc_le_2 - mean ObsValue where quality marker is 0-2
                                inclusive (excludes fill/NaN in either the
                                ObsValue or QC data)
          status            - "ok" or "no_matching_qc_variable"

    Only the ObsValue group and its paired QC group are inspected — every
    other group (ObsError, MetaData, GSI-specific diagnostic groups, etc.)
    is ignored, since none of it matters for this comparison.
    """
    with nc.Dataset(nc_path, mode="r") as ds:
        if OBSVALUE_GROUP not in ds.groups:
            raise ValueError(f"No '{OBSVALUE_GROUP}' group found in {nc_path}")
        obsvalue_group = ds.groups[OBSVALUE_GROUP]

        total_obs = _total_obs_count(ds, obsvalue_group)

        qc_group_name = next((c for c in QC_GROUP_CANDIDATES if c in ds.groups), None)
        qc_group = ds.groups[qc_group_name] if qc_group_name else None

        variables: Dict[str, Dict[str, Any]] = {}

        for var_name, var in obsvalue_group.variables.items():
            obs_data = np.asarray(var[:])
            obs_data = np.ma.filled(obs_data, np.nan) if np.ma.isMaskedArray(obs_data) else obs_data
            if not np.issubdtype(obs_data.dtype, np.number):
                variables[var_name] = {"status": "non_numeric_variable"}
                continue
            obs_fill = _detect_fill_value(var, obs_data)
            obs_valid = _valid_mask(obs_data, obs_fill)

            if qc_group is None or var_name not in qc_group.variables:
                variables[var_name] = {"status": "no_matching_qc_variable", "qc_group": qc_group_name}
                continue

            qc_var = qc_group.variables[var_name]
            qc_data = np.asarray(qc_var[:])
            qc_data = np.ma.filled(qc_data, np.nan) if np.ma.isMaskedArray(qc_data) else qc_data
            qc_fill = _detect_fill_value(qc_var, qc_data)
            qc_valid = _valid_mask(qc_data, qc_fill)

            full_valid = obs_valid & qc_valid
            qc_vals = qc_data.astype("float64")

            le2_mask = full_valid & (qc_vals >= 0) & (qc_vals <= 2)
            gt2_mask = full_valid & (qc_vals > 2)

            n_le2 = int(np.count_nonzero(le2_mask))
            n_gt2 = int(np.count_nonzero(gt2_mask))
            mean_le2 = float(np.mean(obs_data[le2_mask])) if n_le2 > 0 else None

            variables[var_name] = {
                "status": "ok",
                "qc_group": qc_group_name,
                "n_qc_le_2": n_le2,
                "n_qc_gt_2": n_gt2,
                "mean_value_qc_le_2": mean_le2,
            }

        return {
            "total_obs": total_obs,
            "qc_group_used": qc_group_name,
            "variables": variables,
        }
