"""era5._common — shared helpers for the ERA5 zonal-mean generators.

The five ERA5 generators in this package all follow the same recipe:

    open the raw CDS NetCDF  ->  zonal mean over longitude  ->
    12-month groupby climatology  ->  rename ERA5 short names  ->
    provenance stamp  ->  write NetCDF.

The recipe — the year ranges, the zonal-mean / climatology operations — is
the scientific content; this module provides the shared plumbing around it. The shared loop lives in
:func:`zonal_mean_climatology`; per-generator scripts supply only the raw
filenames, the variable renames, and the provenance descriptors.

This module imports ``xarray`` (a lean-core dependency) and reaches into
``core.provenance``; it does NOT require the ``[generators]`` extra at import
time. ``cdsapi`` is only referenced inside the commented download recipes.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import xarray as xr

from ...core import provenance

# Repository root — the git state stamped into generator outputs is that of
# this engine repo (provenance.git_describe walks up from here).
_REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def zonal_mean_climatology(
    raw_file: str,
    out_file: str,
    *,
    generator: str,
    config: Mapping[str, object],
    source: str,
    processing: str,
    period: str,
    renames: Optional[Mapping[str, str]] = None,
) -> xr.Dataset:
    """Run the shared ERA5 load -> zonal-mean -> climatology -> stamp recipe.

    The recipe is:

      1. open the raw CDS NetCDF and load it into memory;
      2. rename ``pressure_level`` -> ``level`` if present;
      3. zonal mean over ``longitude`` (``skipna=True``);
      4. 12-month climatology via ``groupby('valid_time.month').mean``;
      5. rename ERA5 short variable names to the published names;
      6. stamp provenance global attributes;
      7. write the result to ``out_file``.

    Parameters
    ----------
    raw_file, out_file :
        Input raw CDS NetCDF path and output NetCDF path.
    generator :
        Generator name for the provenance stamp, e.g.
        ``"era5.monthly_zonal_o3"``.
    config :
        Resolved generator configuration (recorded in the stamp).
    source, processing, period :
        Free-text provenance descriptors passed through to
        :func:`core.provenance.stamp`.
    renames :
        Optional ``{era5_short_name: published_name}`` mapping; only keys
        present in the climatology dataset are renamed.

    Returns
    -------
    xarray.Dataset
        The stamped climatology dataset (also written to ``out_file``).
    """
    ds = xr.open_dataset(raw_file).load()
    if "pressure_level" in ds.coords:
        ds = ds.rename({"pressure_level": "level"})

    # step 2: zonal mean over longitude
    ds_zonal = ds.mean("longitude", skipna=True)

    # step 3: 12-month climatology -- average over years, grouped by month
    ds_clim = ds_zonal.groupby("valid_time.month").mean("valid_time")

    # rename ERA5 short names to the published variable names
    if renames:
        present = {k: v for k, v in renames.items() if k in ds_clim}
        if present:
            ds_clim = ds_clim.rename(present)

    provenance.stamp(
        ds_clim,
        generator=generator,
        config=dict(config),
        source=source,
        processing=processing,
        period=period,
        repo_dir=_REPO_DIR,
    )

    ds_clim.to_netcdf(out_file)
    return ds_clim
