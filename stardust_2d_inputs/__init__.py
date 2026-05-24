"""stardust_2d_inputs — the input-data engine for the Stardust 2-D model.

Two parts:

* ``core``       — the lean runtime loader + provenance registry. The *only*
                   part imported at simulation runtime. Lean deps only.
* ``generators`` — offline tooling that produces the input files. Heavy /
                   varied deps behind the ``[generators]`` pip extra;
                   populated in a later phase.

The public surface for runtime use is re-exported here from ``core``.
"""

from .core.loader import load
from .core.registry import Registry
from .core import provenance

__all__ = ["load", "Registry", "provenance"]

__version__ = "0.1.0"
