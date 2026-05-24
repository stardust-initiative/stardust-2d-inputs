"""era5 — offline generators for the ERA5 zonal-mean input datasets.

Five generators producing the ERA5 zonal-mean monthly input datasets, each
stamping its output via :func:`core.provenance.stamp`:

================================  ==========================================
module                            published dataset
================================  ==========================================
``monthly_zonal_variables``        Monthly_Zonal_Variables (T/q/u/v/w)
``monthly_zonal_cloud_cover``      Monthly_Zonal_Cloud_Cover
``monthly_zonal_cloud_content``    Monthly_Zonal_Cloud_content
``monthly_zonal_o3``               Monthly_Zonal_o3
``monthly_zonal_srf_params``       Monthly_Zonal_Srf_Params
================================  ==========================================

The first four share the load -> zonal-mean -> climatology -> stamp recipe
in :mod:`era5._common`; ``monthly_zonal_srf_params`` has its own mask-blended
recipe. Each module exposes ``generate(raw_file, out_file)`` and a ``main()``
CLI entry point. Behind the ``[generators]`` pip extra; not imported at
simulation runtime.
"""

from . import (
    monthly_zonal_cloud_content,
    monthly_zonal_cloud_cover,
    monthly_zonal_o3,
    monthly_zonal_srf_params,
    monthly_zonal_variables,
)

__all__ = [
    "monthly_zonal_variables",
    "monthly_zonal_cloud_cover",
    "monthly_zonal_cloud_content",
    "monthly_zonal_o3",
    "monthly_zonal_srf_params",
]
