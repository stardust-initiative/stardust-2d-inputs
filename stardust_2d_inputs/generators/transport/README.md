# generators/transport

Offline generators for the **transport-model driver inputs** — the datasets
that drive the transport-paper model's 2-D stratospheric transport. Behind
the `[generators]` pip extra; not imported at simulation runtime. Each output
is stamped via `core.provenance.stamp()`.

## Generators

### `diffusion_decomp`

Computes the ERA5 model-level (L137) eddy diffusivity tensor and the Pitari
Transformed-Eulerian-Mean decomposition of the eddy transport stream
function. Writes one NetCDF per `(year, month)`:
`diffusion_ml_TEMdecomp_raw_<year>_<month>.nc` (registry key
`eddy_diffusivity`).

Configuration is via the `TemConfig` dataclass, populated from CLI arguments
by `config_from_args`; the output is stamped via `core.provenance.stamp()`
before `to_netcdf`.

This is the **RAW** variant: no postprocessing on the diffusion tensor (no
floor replacement, no eigenvalue constraint, no negative masking), and
`omega_smooth_n = 1` (no omega smoothing). Velocities and stream functions
are polar-filled; the diffusion tensor is not.

A run reads the public ARCO-ERA5 model-level GCS Zarr store
(`gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v{1,2}`) and
takes hours of compute per month. The `[generators]` extra supplies `zarr`,
`gcsfs` and `scipy`; `cupy` is optional (GPU interpolation).

### `tropopause`

Builds the zonal-mean monthly tropopause pressure climatology
`tropopause_ERA5_zonal_mean_<start>_<end>.nc` (registry key `tropopause`),
with the output variable named `tropopause_pressure`. For each
`(year, month)` and latitude it takes the higher-pressure (lower-altitude) of
the dynamical and WMO-1st-thermal tropopause estimates.

The source is the Hoffmann & Spang Reanalysis Tropopause Data Repository
(DOI [10.26165/JUELICH-DATA/UBNGI2](https://doi.org/10.26165/JUELICH-DATA/UBNGI2)),
`era5low` zonal-mean subset. A run reads the downloaded `.tab` files locally;
no network or heavy compute is required.

Configuration is via the `TropopauseConfig` dataclass (required `data_dir`,
plus year range), populated from CLI arguments by `config_from_args`; the
output is stamped via `core.provenance.stamp()` before `to_netcdf`.

## Usage

```bash
# diffusion_decomp — single month
python -m stardust_2d_inputs.generators.transport.diffusion_decomp \
    --years 2010 2010 --months 1 1 --outdir ./transport_out

# diffusion_decomp — full multi-year run
python -m stardust_2d_inputs.generators.transport.diffusion_decomp \
    --years 2008 2017 --months 1 12 --workers 4

# tropopause — full 2008-2017 climatology
python -m stardust_2d_inputs.generators.transport.tropopause \
    --data-dir /path/to/era5low_tab_files --years 2008 2017
```

Or from Python:

```python
from stardust_2d_inputs.generators.transport.diffusion_decomp import TemConfig, run
run(TemConfig(year_start=2010, year_end=2010, month_start=1, month_end=1))

from stardust_2d_inputs.generators.transport.tropopause import TropopauseConfig, run
run(TropopauseConfig(data_dir="/path/to/era5low_tab_files",
                     year_start=2008, year_end=2017))
```
