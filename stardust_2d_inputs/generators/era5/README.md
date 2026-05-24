# generators/era5

Offline generators for the ERA5 zonal-mean input datasets. Behind the
`[generators]` pip extra; not imported at simulation runtime. Each generator
stamps its NetCDF output via `core.provenance.stamp()`.

## Generators

| Module | Published dataset | Recipe |
| --- | --- | --- |
| `monthly_zonal_variables` | `Monthly_Zonal_Variables` (T/q/u/v/w) | zonal-mean climatology, 2008-2017 |
| `monthly_zonal_cloud_cover` | `Monthly_Zonal_Cloud_Cover` | zonal-mean climatology, 2009-2018 |
| `monthly_zonal_cloud_content` | `Monthly_Zonal_Cloud_content` | zonal-mean climatology, 2008-2017 |
| `monthly_zonal_o3` | `Monthly_Zonal_o3` | zonal mean, 12 monthly means of 2008 (not a climatology) |
| `monthly_zonal_srf_params` | `Monthly_Zonal_Srf_Params` | mask-blended surface T / humidity, zonal mean, 2008-2017 |

These are **general-provenance recipes**, not bit-identical regenerators: they
reproduce equivalent datasets and record how each was made. The commented
`cdsapi` request block at the top of each module is the provenance recipe for
the raw download step.

The first four share the load -> zonal-mean -> 12-month climatology -> stamp
loop in `_common.zonal_mean_climatology`. `monthly_zonal_srf_params` has its
own recipe (integrated surface temperature blended from skin / SST / sea-ice
fields by land-sea and sea-ice masks; surface and 2 m specific humidity from a
saturation-vapour-pressure relation).

## Usage

A run needs a Copernicus CDS account configured for `cdsapi`. First retrieve
the raw NetCDF with the commented `cdsapi` request at the top of the chosen
module, then run the generator:

```bash
python -m stardust_2d_inputs.generators.era5.monthly_zonal_o3 \
    --raw-file ./era5_2008_2017_o3.nc \
    --out-file ./Monthly_Zonal_o3_2008_2017.nc
```

Or from Python:

```python
from stardust_2d_inputs.generators.era5 import monthly_zonal_o3
monthly_zonal_o3.generate("./era5_2008_2017_o3.nc", "./Monthly_Zonal_o3_2008_2017.nc")
```
