# generators/optical

Offline generators for the aerosol optical-table input datasets. Behind the
`[generators]` pip extra; not imported at simulation runtime. Each generator
stamps its NetCDF output via `core.provenance.stamp()`.

## Pipeline

A two-stage pipeline produces, per material, the climlab-format RRTMG optical
table:

- **Stage 1** — per-material refractive-index generation. Each material module
  builds the complex refractive index `m(lambda) = n + i*k` on a log-spaced
  wavelength grid from its published spectroscopic sources, and writes an
  intermediate `<material>_optical.csv`.
- **Stage 2** — the shared Mie + RRTMG-band machinery in `_common.py`. The
  intermediate CSV is loaded into a `Particle`, `miepython` computes the Mie
  extinction / scattering / absorption efficiencies per radius, and the
  spectral quantities are Planck-weighted into the RRTMG shortwave (14-band)
  and longwave (16-band) tables. `RadiationTransferTableRRTMG` drives the
  radius loop and writes the NetCDF.

## Generators

| Material module | Output climlab table | Refractive-index source |
| --- | --- | --- |
| `silica` | `stardust_particles_silica_climlab.nc` | Kitamura et al. (2007) oscillator model + sampled n/k below 8 um |
| `sulfate` | `stardust_particles_H2SO4_75_climlab.nc` | stitched 75% H2SO4 index (Hummel 1988 / Ferraro 2014 / Gosse 1997) |
| `calcite` | `stardust_particles_calcite_AvgRay_climlab.nc` | Long et al. (1993) ray-averaged Lorentz-oscillator model |

## Raw inputs

Carried under `data/<material>/`, restructured from the source repository.
Only the final refractive-index sources actually used by the generators are
carried (no OCR intermediates, no plot images):

- `data/silica/` — sampled n/k CSVs (`n_0-1um.csv`, `n_1-15um.csv`,
  `k_0-1um.csv`, `k_1-15um.csv`) plus the `reference.txt` citation.
- `data/sulfate/` — `ref3_sulfate_table_h2so4_75percent.csv` (background
  table), `sulfate_data_s1.txt` (Ferraro 2014 index), and
  `sulfate_absorption_ref1.csv` (Gosse 1997 absorption).
- Calcite needs no raw inputs — its Stage-1 model is purely analytic.

## Usage

```bash
python -m stardust_2d_inputs.generators.optical.silica \
    --optical-csv ./silica_optical.csv \
    --out-file ./stardust_particles_silica_climlab.nc
```

Or from Python:

```python
from stardust_2d_inputs.generators.optical import silica
silica.generate("./silica_optical.csv", "./stardust_particles_silica_climlab.nc")
```

A full table run is Mie-heavy (the radius loop calls `miepython` per radius
and per zenith angle); `--nr` lowers the radius-sample count for a quick
smoke run.
