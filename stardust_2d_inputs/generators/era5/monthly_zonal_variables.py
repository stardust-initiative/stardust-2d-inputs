"""era5.monthly_zonal_variables — Monthly_Zonal_Variables ERA5 generator.

General-provenance recipe (not a bit-identical regenerator). The download
recipe and the zonal-mean / climatology operations run through
:func:`era5._common.zonal_mean_climatology`, with output stamped via
:func:`core.provenance.stamp`.

SCOPE -- read this. The published Monthly_Zonal_Variables_2008_2017.nc has
39 variables:
  * T, q, u, v, w          -- zonal-mean state fields (this script)
  * GP, h                  -- geopotential and moist static energy
  * {v,w}{T,q,GP,h}_{mean,stat,trans,total}  -- meridional/vertical
                              eddy-flux decompositions (32 fields)

The transport paper's SARF and 2-D transport runs use ONLY the T/q/u/v/w
zonal means -- T and q seed the ERA5 reference state, T also sets the Kzz
density floor in get_atm_data. This script reproduces those five fields. `q` is ERA5 native specific
humidity (this generator downloads `specific_humidity` directly).

The geopotential, MSE and eddy-flux fields are produced by a separate
eddy-flux / TEM decomposition pipeline and are not reproduced here -- those
fields are off the paper's critical path.

The climatology is built from the ERA5 monthly means for 2008-2017.
"""

from __future__ import annotations

import argparse

from ._common import zonal_mean_climatology

GENERATOR = "era5.monthly_zonal_variables"

# --- step 1: ERA5 download (Copernicus Climate Data Store) -------------
# The raw file below was retrieved with this cdsapi request:
#
# import cdsapi
# dataset = "reanalysis-era5-pressure-levels-monthly-means"
# request = {
#     "product_type": ["monthly_averaged_reanalysis"],
#     "variable": [
#         "temperature",
#         "specific_humidity",
#         "u_component_of_wind",
#         "v_component_of_wind",
#         "vertical_velocity"],
#     "pressure_level": [
#         "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
#         "100", "125", "150", "175", "200", "225", "250", "300", "350",
#         "400", "450", "500", "550", "600", "650", "700", "750", "775",
#         "800", "825", "850", "875", "900", "925", "950", "975", "1000"],
#     "year": ["2008", "2009", "2010", "2011", "2012",
#              "2013", "2014", "2015", "2016", "2017"],
#     "month": ["01", "02", "03", "04", "05", "06",
#               "07", "08", "09", "10", "11", "12"],
#     "time": ["00:00"],
#     "data_format": "netcdf",
#     "download_format": "unarchived",
# }
# cdsapi.Client().retrieve(dataset, request).download()

DEFAULT_RAW_FILE = "./era5_2008_2017_variables.nc"
DEFAULT_OUT_FILE = "./Monthly_Zonal_Variables_2008_2017.nc"

SOURCE = "ERA5 reanalysis monthly means on pressure levels (Copernicus CDS)"
PROCESSING = "zonal mean over longitude; 12-month climatology over 2008-2017"
PERIOD = "2008-2017"

# ERA5 short name `t` -> the published variable name `T`
# (q, u, v, w already match the published names).
RENAMES = {"t": "T"}


def generate(raw_file: str = DEFAULT_RAW_FILE, out_file: str = DEFAULT_OUT_FILE):
    """Build the Monthly_Zonal_Variables climatology and stamp its provenance.

    NB: this writes the T/q/u/v/w zonal means only. The eddy-flux / GP / h
    fields of the published file come from the separate pipeline noted in
    the module docstring and are not produced here.
    """
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
