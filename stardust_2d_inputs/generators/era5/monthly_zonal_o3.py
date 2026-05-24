"""era5.monthly_zonal_o3 — Monthly_Zonal_o3 ERA5 generator.

General-provenance recipe (not a bit-identical regenerator): download the
ERA5 monthly-mean ozone mass mixing ratio on pressure levels from the
Copernicus Climate Data Store and zonally average. The zonal-mean /
climatology operations run through
:func:`era5._common.zonal_mean_climatology`, with output stamped via
:func:`core.provenance.stamp`.

The `_2008_2017` filename suffix is historical: the published file is not a
multi-year climatology — it holds the 12 monthly means of the single year
2008. This script therefore downloads month means for 2008 only;
the `groupby('valid_time.month')` step is then a no-op pass-through (one year
per month). The published file is trimmed to the 22 upper pressure levels
that carry the ozone layer; the model interpolates onto its own grid, so the
level subset is not material -- this script downloads the full 37-level set.
"""

from __future__ import annotations

import argparse

from ._common import zonal_mean_climatology

GENERATOR = "era5.monthly_zonal_o3"

# --- step 1: ERA5 download (Copernicus Climate Data Store) -------------
# The raw file below was retrieved with this cdsapi request:
#
# import cdsapi
# dataset = "reanalysis-era5-pressure-levels-monthly-means"
# request = {
#     "product_type": ["monthly_averaged_reanalysis"],
#     "variable": ["ozone_mass_mixing_ratio"],
#     "pressure_level": [
#         "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
#         "100", "125", "150", "175", "200", "225", "250", "300", "350",
#         "400", "450", "500", "550", "600", "650", "700", "750", "775",
#         "800", "825", "850", "875", "900", "925", "950", "975", "1000"],
#     "year": ["2008"],
#     "month": ["01", "02", "03", "04", "05", "06",
#               "07", "08", "09", "10", "11", "12"],
#     "time": ["00:00"],
#     "data_format": "netcdf",
#     "download_format": "unarchived",
# }
# cdsapi.Client().retrieve(dataset, request).download()

DEFAULT_RAW_FILE = "./era5_2008_2017_o3.nc"
DEFAULT_OUT_FILE = "./Monthly_Zonal_o3_2008_2017.nc"

SOURCE = (
    "ERA5 monthly-mean ozone mass mixing ratio on pressure levels " "(Copernicus CDS)"
)
# With a single year (2008) downloaded the groupby('valid_time.month') step
# is a pass-through, NOT a multi-year average -- the output holds the 12
# monthly means of 2008.
PROCESSING = (
    "zonal mean over longitude; the 12 monthly means of the single year "
    "2008 (NOT a multi-year climatology)"
)
PERIOD = "2008"

# ERA5 short name `o3` matches the published variable name -- no renames.
RENAMES = None


def generate(raw_file: str = DEFAULT_RAW_FILE, out_file: str = DEFAULT_OUT_FILE):
    """Build the Monthly_Zonal_o3 dataset (12 monthly means of 2008) and stamp it."""
    return zonal_mean_climatology(
        raw_file,
        out_file,
        generator=GENERATOR,
        config={"raw_file": raw_file, "out_file": out_file},
        source=SOURCE,
        processing=PROCESSING,
        period=PERIOD,
        renames=RENAMES,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-file",
        default=DEFAULT_RAW_FILE,
        help="raw ERA5 CDS NetCDF (default: %(default)s)",
    )
    parser.add_argument(
        "--out-file",
        default=DEFAULT_OUT_FILE,
        help="output NetCDF path (default: %(default)s)",
    )
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    generate(args.raw_file, args.out_file)


if __name__ == "__main__":
    main()
