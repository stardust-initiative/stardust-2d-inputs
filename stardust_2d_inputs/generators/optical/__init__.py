"""optical — offline generators for the aerosol optical-table input datasets.

A two-stage pipeline produces, per material, the climlab-format RRTMG optical
table (``stardust_particles_*_climlab.nc``):

  Stage 1 — per-material refractive-index generation: build the complex index
            ``m(lambda) = n + i*k`` from published spectroscopic sources.
  Stage 2 — the Mie + RRTMG-band machinery in :mod:`optical._common`:
            ``miepython`` Mie efficiencies Planck-weighted into the RRTMG
            shortwave (14-band) and longwave (16-band) tables.

================  ====================================================
material module   output climlab table
================  ====================================================
``silica``        ``stardust_particles_silica_climlab.nc``
``sulfate``       ``stardust_particles_H2SO4_75_climlab.nc``
``calcite``       ``stardust_particles_calcite_AvgRay_climlab.nc``
================  ====================================================

Each material module exposes ``generate()`` / ``main()`` and is driven by a
:class:`optical._common.OpticalConfig`; each stamps its NetCDF output via
:func:`core.provenance.stamp`. Behind the ``[generators]`` pip extra
(``miepython``, ``scipy``, ``pandas``); not imported at simulation runtime.
"""

from . import calcite, silica, sulfate

# Material registry: the materials this package generates optical tables for.
MATERIALS = {
    "silica": silica,
    "sulfate": sulfate,
    "calcite": calcite,
}

__all__ = ["silica", "sulfate", "calcite", "MATERIALS"]
