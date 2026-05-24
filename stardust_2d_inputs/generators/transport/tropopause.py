#!/usr/bin/env python3
"""transport.tropopause — zonal-mean monthly tropopause-pressure generator.

Builds a zonal-mean monthly tropopause-pressure climatology from the
Hoffmann & Spang Reanalysis Tropopause Data Repository. For each
``(year, month)`` and latitude, the tropopause pressure is taken as the
per-latitude **maximum** of the dynamical and WMO-1st-thermal tropopause
pressures (i.e. the lower-altitude / higher-pressure of the two estimates).
The output is one stamped NetCDF over ``(year, month, latitude)``.

Source
------
Hoffmann, L. and R. Spang, *Reanalysis Tropopause Data Repository*,
https://doi.org/10.26165/JUELICH-DATA/UBNGI2, Jülich DATA, V1, 2021. The
specific subset used here is the ``era5low`` zonal-mean product:
https://datapub.fz-juelich.de/slcs/tropopause/data/projects/zonal_mean/era5low/

Method
------
For each year and month, the ``era5low_dyn_<year>_<mm>.tab`` (dynamical) and
``era5low_wmo_1st_<year>_<mm>.tab`` (WMO first thermal) zonal-mean tables are
read, their latitude axes checked for agreement, and the per-latitude
elementwise maximum of the two pressure columns is taken. The result is
stacked over ``(year, month, latitude)``.

Configuration is via :class:`TropopauseConfig` (CLI-populated); the output is
stamped via :func:`core.provenance.stamp` before writing. The output
DataArray is named ``tropopause_pressure`` with CF ``long_name`` / ``units``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...core import provenance

GENERATOR = "transport.tropopause"

# Repository root for the provenance git-state stamp.
_REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# ═══════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════


@dataclass
class TropopauseConfig:
    """Resolved configuration for a tropopause-climatology run.

    Field values are populated from CLI arguments by :func:`config_from_args`.
    ``data_dir`` is required: it is the local directory holding the downloaded
    ``era5low_{dyn,wmo_1st}_<year>_<mm>.tab`` tables.
    """

    data_dir: str
    year_start: int = 2008
    year_end: int = 2017

    def years(self) -> list[int]:
        """The years this run covers (inclusive)."""
        return list(range(self.year_start, self.year_end + 1))

    def stamp_config(self) -> dict:
        """The config dict recorded in the provenance stamp."""
        return {
            "data_dir": self.data_dir,
            "year_start": self.year_start,
            "year_end": self.year_end,
        }


# ═══════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════


def load_file_skip_header(filepath: str) -> np.ndarray:
    """Load a ``.tab`` table, skipping the comment header lines.

    Header lines are those that do not begin with a digit; the remaining
    numeric rows are parsed with :func:`numpy.genfromtxt`. Columns are
    ``[time, lat, height, pressure, ...]``.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    data_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
    return np.genfromtxt(data_lines)


# ═══════════════════════════════════════════════════
#  Core build
# ═══════════════════════════════════════════════════


def build_dataarray(cfg: TropopauseConfig) -> xr.DataArray:
    """Build the ``tropopause_pressure`` DataArray for ``cfg``.

    Reads the dynamical and WMO-1st zonal-mean tables for every
    ``(year, month)`` in ``cfg``, takes the per-latitude max of the two
    tropopause pressures, and stacks the result over
    ``(year, month, latitude)``. The returned DataArray is named
    ``tropopause_pressure`` with CF ``long_name`` / ``units`` attrs.
    """
    months = [f"{m:02d}" for m in range(1, 13)]
    years = cfg.years()

    all_years_data = []
    latitude = None

    for year in years:
        print(f"Processing year {year}...")
        monthly_trop_p = []

        for mm in months:
            dyn_file = os.path.join(cfg.data_dir, f"era5low_dyn_{year}_{mm}.tab")
            wmo_file = os.path.join(cfg.data_dir, f"era5low_wmo_1st_{year}_{mm}.tab")

            try:
                dyn_data = load_file_skip_header(dyn_file)
                wmo_data = load_file_skip_header(wmo_file)
            except FileNotFoundError as e:
                print(f"Warning: File not found - {e}")
                continue

            # Columns: [time, lat, height, pressure, ...]
            lat_dyn, p_dyn = dyn_data[:, 1], dyn_data[:, 3]
            lat_wmo, p_wmo = wmo_data[:, 1], wmo_data[:, 3]

            if not np.allclose(lat_dyn, lat_wmo, atol=0.1):
                raise ValueError(f"Latitude mismatch in year {year}, month {mm}")

            if latitude is None:
                latitude = lat_dyn

            # Higher pressure = lower altitude = max(pressure).
            combined_p = np.maximum(p_dyn, p_wmo)
            monthly_trop_p.append(combined_p)

        if monthly_trop_p:
            all_years_data.append(np.stack(monthly_trop_p))

    if not all_years_data:
        raise RuntimeError(
            f"No tropopause tables found under {cfg.data_dir!r} for "
            f"years {cfg.year_start}-{cfg.year_end}"
        )

    # Stack all years: shape = (n_years, 12, N_lat).
    all_years_array = np.stack(all_years_data)

    da = xr.DataArray(
        all_years_array,
        dims=["year", "month", "latitude"],
        coords={
            "year": np.array(years),
            "month": np.arange(1, 13),
            "latitude": latitude,
        },
        name="tropopause_pressure",
        attrs={
            "long_name": "Tropopause Pressure",
            "units": "hPa",
            "description": (
                f"Combined WMO/dynamical tropopause pressure "
                f"{cfg.year_start}-{cfg.year_end}, per year and month"
            ),
            "method": "Maximum of WMO and dynamical tropopause pressures",
        },
    )
    return da


# ═══════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════


def run(cfg: TropopauseConfig, output_file: str | None = None) -> str:
    """Build the climatology, stamp it, and write it to NetCDF.

    Returns the path of the written file. ``output_file`` defaults to
    ``tropopause_ERA5_zonal_mean_<start>_<end>.nc`` in the current directory.
    """
    da = build_dataarray(cfg)

    print("\nDataArray created:")
    print(da)

    # Convert to a Dataset so the provenance stamp lands on global attrs while
    # the data variable keeps its CF attrs and its name (tropopause_pressure).
    ds = da.to_dataset()

    provenance.stamp(
        ds,
        generator=GENERATOR,
        config=cfg.stamp_config(),
        source=(
            "Hoffmann & Spang Reanalysis Tropopause Data Repository "
            "(DOI 10.26165/JUELICH-DATA/UBNGI2), era5low zonal-mean subset"
        ),
        processing=(
            "per-latitude maximum of dynamical and WMO-1st tropopause "
            "pressures, by year and month"
        ),
        period=f"{cfg.year_start}-{cfg.year_end}",
        repo_dir=_REPO_DIR,
    )

    if output_file is None:
        output_file = f"tropopause_ERA5_zonal_mean_{cfg.year_start}_{cfg.year_end}.nc"
    ds.to_netcdf(output_file)
    print(f"\nSaved to {output_file}")
    return output_file


# ═══════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zonal-mean monthly tropopause-pressure climatology "
        "from the Hoffmann & Spang era5low repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full 2008-2017 climatology:
      python -m stardust_2d_inputs.generators.transport.tropopause \\
          --data-dir /path/to/era5low_tab_files --years 2008 2017
""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="local directory holding the era5low_{dyn,wmo_1st}_<year>_<mm>.tab tables",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=(2008, 2017),
        metavar=("START", "END"),
        help="inclusive year range (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output NetCDF path (default: tropopause_ERA5_zonal_mean_<start>_<end>.nc)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TropopauseConfig:
    """Build a :class:`TropopauseConfig` from parsed CLI arguments."""
    year_start, year_end = args.years
    return TropopauseConfig(
        data_dir=args.data_dir,
        year_start=year_start,
        year_end=year_end,
    )


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    run(config_from_args(args), output_file=args.output)


if __name__ == "__main__":
    main()
