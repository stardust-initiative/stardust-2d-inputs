"""era5.monthly_zonal_srf_params — Monthly_Zonal_Srf_Params ERA5 generator.

General-provenance recipe (not a bit-identical regenerator).

This generator does not use the shared ``zonal_mean_climatology`` helper:
its recipe is not a plain zonal-mean climatology. It derives an integrated
surface temperature ``ts`` (skin / sea-surface / sea-ice blended by
land-sea and sea-ice masks) and the surface / 2 m specific humidities
``qs`` / ``q2m`` from a saturation-vapour-pressure relation, then zonally
averages. The output is stamped via :func:`core.provenance.stamp`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

from ...core import provenance

GENERATOR = "era5.monthly_zonal_srf_params"

# --- step 1: ERA5 download (Copernicus Climate Data Store) -------------
# The raw file below was retrieved with this cdsapi request:
#
# import cdsapi
# dataset = "reanalysis-era5-single-levels-monthly-means"
# request = {
#     "product_type": ["monthly_averaged_reanalysis"],
#     "variable": [
#         "2m_dewpoint_temperature",
#         "2m_temperature",
#         "sea_surface_temperature",
#         "surface_pressure",
#         "ice_temperature_layer_1",
#         "skin_temperature",
#         "land_sea_mask",
#         "sea_ice_cover"
#     ],
#     "year": [
#         "2008", "2009", "2010",
#         "2011", "2012", "2013",
#         "2014", "2015", "2016",
#         "2017"
#     ],
#     "month": [
#         "01", "02", "03",
#         "04", "05", "06",
#         "07", "08", "09",
#         "10", "11", "12"
#     ],
#     "time": ["00:00"],
#     "data_format": "netcdf",
#     "download_format": "unarchived"
# }
# client = cdsapi.Client()
# client.retrieve(dataset, request).download()

DEFAULT_RAW_FILE = "./era5_2008_2017_surface_parameters.nc"
DEFAULT_OUT_FILE = "./Monthly_Zonal_Srf_Params_2008_2017.nc"

SOURCE = "ERA5 reanalysis monthly means on single levels (Copernicus CDS)"
PROCESSING = (
    "mask-blended integrated surface temperature (skin/SST/sea-ice); "
    "surface and 2 m specific humidity from a saturation-vapour-pressure "
    "relation; zonal mean over longitude"
)
PERIOD = "2008-2017"

# Repository root for the provenance git-state stamp.
_REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def generate(raw_file: str = DEFAULT_RAW_FILE, out_file: str = DEFAULT_OUT_FILE):
    """Build the Monthly_Zonal_Srf_Params dataset and stamp its provenance.

    The numeric recipe below — saturation-vapour-pressure / specific-humidity
    functions, the land-sea / sea-ice mask blending, and the zonal mean — is
    the established surface-parameter recipe.
    """
    # --- surface-parameter recipe -----------------------------------------
    wvmw_ratio = 0.622  # ratio of molecular weight of water vapor to dry air
    t0c = 273.15  # K
    sat_pres_func = (
        lambda t_k: 6.112 * np.exp(17.67 * (t_k - t0c) / (t_k - t0c + 243.5)) * 1e2
    )
    q_func = lambda p, e: wvmw_ratio * e / (p - (1 - wvmw_ratio) * e)  # Te

    ds = xr.open_dataset(raw_file)

    # Basic masks
    is_ocean = (ds["lsm"] < 0.5) & (ds["siconc"] < 0.15)  # no sea ice
    is_land = (ds["lsm"] >= 0.5) & (ds["siconc"] < 0.15)
    is_ice = ds["siconc"] >= 0.15

    # Initialize final field
    Ts = ds["skt"]
    # Assign based on regions
    Ts = xr.where(is_ocean, ds["sst"], Ts)
    Ts = xr.where(is_ice, ds["istl1"], Ts)
    Ts.name = "ts"

    T1 = ds["t2m"]
    Td1_c = ds["d2m"] - 273.15  # dewpoint temperature at 2m in units of C
    q2m = q_func(ds["sp"], sat_pres_func(ds["d2m"]))
    q2m.name = "q2m"
    qs = q_func(ds["sp"], sat_pres_func(Ts))
    qs.name = "qs"

    ds_add_zonal = xr.Dataset({da.name: da for da in [Ts, q2m, qs]}).mean(
        "longitude", skipna=True
    )
    ds_zonal = ds[["d2m", "t2m", "sst", "skt", "istl1"]].mean("longitude", skipna=True)
    ds_zonal = xr.merge([ds_zonal, ds_add_zonal])
    # --- end surface-parameter recipe -------------------------------------

    provenance.stamp(
        ds_zonal,
        generator=GENERATOR,
        config={"raw_file": raw_file, "out_file": out_file},
        source=SOURCE,
        processing=PROCESSING,
        period=PERIOD,
        repo_dir=_REPO_DIR,
    )

    ds_zonal.to_netcdf(out_file)
    return ds_zonal


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
