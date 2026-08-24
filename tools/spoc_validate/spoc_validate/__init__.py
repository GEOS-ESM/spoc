"""
spoc_validate
=============

Tools for validating new SPOC (https://github.com/GEOS-ESM/spoc) tags by:

  1. grab_and_convert  - pull config/scripts for a given tag, grab BUFR (and
                          optionally ncdiag) inputs, run the bufr->IODA
                          conversion, and tag the output with provenance.
  2. compare_files      - diff SPOC IODA output against GSI-ncdiag-derived
                          IODA output (fields, fill values, means).
  3. hofx_swell         - create SPOC/bufr-query and ncdiag swell hofx
                          experiments per JEDI obs space, and generate a
                          post-launch activation script for each.

Stage 4 (analyzing hofx output) lives in notebooks/04_analyze_hofx_output.ipynb
since it's meant to be run interactively in JupyterLab.
"""

__version__ = "0.1.0"
