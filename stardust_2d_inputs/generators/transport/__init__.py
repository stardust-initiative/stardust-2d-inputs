"""transport — offline generators for the transport-model driver inputs.

This bucket holds the offline generators whose outputs drive the
transport-paper model's 2-D stratospheric transport:

* :mod:`transport.diffusion_decomp` computes the ERA5 model-level eddy
  diffusivity tensor and the Pitari Transformed-Eulerian-Mean decomposition
  of the eddy transport stream function, writing one stamped NetCDF per
  ``(year, month)``.
* :mod:`transport.tropopause` builds the zonal-mean monthly tropopause
  pressure climatology (max of dynamical and WMO-1st tropopause) from the
  Hoffmann & Spang reanalysis tropopause repository.

Behind the ``[generators]`` pip extra; not imported at simulation runtime.
"""

from importlib import import_module

__all__ = ["diffusion_decomp", "tropopause", "isentropic_subset"]


def __getattr__(name):
    # Field selection needs only the core NetCDF dependencies; avoid importing
    # unrelated trajectory-generator dependencies (and GPU probes) eagerly.
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
