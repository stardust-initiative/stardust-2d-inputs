"""era5.monthly_zonal_cloud_cover — Monthly_Zonal_Cloud_Cover ERA5 generator.

General-provenance recipe (not a bit-identical regenerator): download the
ERA5 monthly-mean fraction-of-cloud-cover field on pressure levels from the
Copernicus Climate Data Store, zonally average, and reduce to a 12-month
climatology over 2009-2018. The zonal-mean / climatology operations run
through :func:`era5._common.zonal_mean_climatology`, with output stamped
via :func:`core.provenance.stamp`.

Provenance note: the published Monthly_Zonal_Cloud_Cover file was produced
from a separate GRIB-sourced historic zonal-mean cloud dataset. The recipe
below is a general-provenance reproduction via the CDS — it yields an
equivalent 2009-2018 climatology, with cloud fraction agreeing to ~1e-2.

The `_2008_2017` filename suffix is historical and does not match the
climatology period; the climatology covers the ERA5 monthly means for
2009-2018.
"""

from __future__ import annotations

import argparse

from ._common import zonal_mean_climatology

GENERATOR = "era5.monthly_zonal_cloud_cover"

# --- step 1: ERA5 download (Copernicus Climate Data Store) -------------
# The raw file below was retrieved with this cdsapi request:
#
# import cdsapi
# dataset = "reanalysis-era5-pressure-levels-monthly-means"
# request = {
#     "product_type": ["monthly_averaged_reanalysis"],
#     "variable": ["fraction_of_cloud_cover"],
#     "pressure_level": [
#         "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
#         "100", "125", "150", "175", "200", "225", "250", "300", "350",
#         "400", "450", "500", "550", "600", "650", "700", "750", "775",
#         "800", "825", "850", "875", "900", "925", "950", "975", "1000"],
#     "year": ["2009", "2010", "2011", "2012", "2013",
#              "2014", "2015", "2016", "2017", "2018"],
#     "month": ["01", "02", "03", "04", "05", "06",
#               "07", "08", "09", "10", "11", "12"],
#     "time": ["00:00"],
#     "data_format": "netcdf",
#     "download_format": "unarchived",
# }
# cdsapi.Client().retrieve(dataset, request).download()

DEFAULT_RAW_FILE = "./era5_2008_2017_cloud_cover.nc"
DEFAULT_OUT_FILE = "./Monthly_Zonal_Cloud_Cover_2008_2017.nc"

SOURCE = (
    "ERA5 monthly-mean fraction-of-cloud-cover on pressure levels " "(Copernicus CDS)"
)
PROCESSING = "zonal mean over longitude; 12-month climatology over 2009-2018"
PERIOD = "2009-2018"

# ERA5 short name `cc` -> the published variable name `cloud_cover`
RENAMES = {"cc": "cloud_cover"}


def generate(raw_file: str = DEFAULT_RAW_FILE, out_file: str = DEFAULT_OUT_FILE):
    """Build the Monthly_Zonal_Cloud_Cover climatology and stamp provenance."""
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
