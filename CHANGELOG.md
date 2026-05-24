# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-24

Initial public release.

### Added

- Lean runtime `core/` — content-addressed loader with provenance
  registry, pluggable local / GCS / Zenodo backends, provenance stamping
  on loaded artifacts, and a small configuration layer.
- Offline generators:
  - `stardust_2d_inputs.generators.era5` — ERA5 zonal-mean monthly fields
    (temperature, specific humidity, cloud cover, cloud water content,
    ozone, surface parameters, surface shortwave balance).
  - `stardust_2d_inputs.generators.transport` — transport drivers
    (transformed-Eulerian-mean residual circulation + eddy-diffusivity
    tensor derived from ERA5; zonal-mean tropopause climatology).
  - `stardust_2d_inputs.generators.optical` — RRTMG-banded aerosol
    optical-property tables (silica, sulfate at 75 % H₂SO₄, calcite).
- The `transport-paper-v1` release manifest pinning the 11 input datasets
  for Lederer et al. (2026), backed by the public Zenodo deposit
  [10.5281/zenodo.20271742](https://doi.org/10.5281/zenodo.20271742).
