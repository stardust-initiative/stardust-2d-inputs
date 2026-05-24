"""core — the lean runtime layer of the input-data engine.

Modules
-------
provenance : generator-output stamp helper (release-workflow standard §5a).
registry   : load the provenance registry; resolve (key, version) -> entry.
backends   : storage backends (local / GCS / Zenodo) with a content-addressed
             local cache.
loader     : the public ``load(key, ...)`` entry point.
config     : config loading + backend selection.

``core`` must stay lean — no generator dependencies (cdsapi, zarr, scipy).
The GCS backend's ``gcsfs`` dependency is imported lazily, so importing
``core`` itself never requires the ``[gcs]`` extra.
"""

from . import provenance, registry, backends, loader, config

__all__ = ["provenance", "registry", "backends", "loader", "config"]
