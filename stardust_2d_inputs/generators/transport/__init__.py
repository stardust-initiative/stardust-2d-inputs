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

from . import diffusion_decomp, tropopause

__all__ = ["diffusion_decomp", "tropopause"]
