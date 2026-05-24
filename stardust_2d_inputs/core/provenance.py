"""provenance — generator-output stamping (release-workflow standard §5a).

Every generator stamps the NetCDF file it produces, via the shared
``stamp()`` helper here, with a set of global attributes that record *how*
the file was made: which generator, the engine-repo git state at the time,
the resolved configuration, a creation timestamp, and free-text
source / processing / period descriptors.

This module is part of the lean ``core`` — no generator dependencies. It is
called by the generators (Phase B2) but defined here so the runtime side can
also *read* a stamp back off a dataset.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from typing import Any, Mapping, Optional

# Global-attribute names written by stamp(). Kept as module constants so the
# loader / tests can refer to them without hard-coding strings.
ATTR_GENERATOR = "generator"
ATTR_GENERATOR_VERSION = "generator_version"
ATTR_GENERATOR_CONFIG = "generator_config"
ATTR_CREATED = "created"
ATTR_SOURCE = "source"
ATTR_PROCESSING = "processing"
ATTR_PERIOD = "period"

STAMP_ATTRS = (
    ATTR_GENERATOR,
    ATTR_GENERATOR_VERSION,
    ATTR_GENERATOR_CONFIG,
    ATTR_CREATED,
    ATTR_SOURCE,
    ATTR_PROCESSING,
    ATTR_PERIOD,
)


def git_describe(repo_dir: Optional[str] = None) -> str:
    """Return a human-readable git version string for ``repo_dir``.

    Format: ``<git describe --tags --always --dirty>`` followed by the short
    SHA and a clean / dirty flag, e.g. ``v0.2.1-3-gabc1234-dirty``. If the
    directory is not a git repository (or git is unavailable) the string
    ``"unknown"`` is returned rather than raising — a generator must still be
    able to stamp its output.

    Per standard §5a, official outputs are produced from a clean git state;
    a dirty tree is *recorded and flagged* (the ``-dirty`` suffix) rather
    than rejected here.
    """
    repo_dir = repo_dir or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def stamp(
    dataset: "xarray.Dataset",  # noqa: F821  (xarray imported lazily by caller)
    generator: str,
    config: Mapping[str, Any],
    *,
    source: str = "",
    processing: str = "",
    period: str = "",
    repo_dir: Optional[str] = None,
    created: Optional[str] = None,
) -> "xarray.Dataset":  # noqa: F821
    """Write the §5a provenance global attributes onto ``dataset``.

    Parameters
    ----------
    dataset :
        The xarray ``Dataset`` produced by a generator. Mutated in place
        (its ``.attrs`` updated) and also returned for convenience.
    generator :
        Name of the generator that produced the file, e.g.
        ``"era5.monthly_zonal_variables"``.
    config :
        The fully-resolved generator configuration. Serialized to a JSON
        string in the ``generator_config`` attribute (NetCDF attrs must be
        scalars / strings, not nested objects).
    source, processing, period :
        Free-text provenance descriptors: where the raw data came from, what
        processing was applied, and the time period covered.
    repo_dir :
        Directory of the generator's git repository, used for the version
        string. Defaults to the current working directory.
    created :
        ISO-8601 creation timestamp. Defaults to ``now`` in UTC.

    Returns
    -------
    xarray.Dataset
        The same ``dataset`` object, with global attributes added.
    """
    if created is None:
        created = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

    attrs = {
        ATTR_GENERATOR: str(generator),
        ATTR_GENERATOR_VERSION: git_describe(repo_dir),
        ATTR_GENERATOR_CONFIG: json.dumps(dict(config), sort_keys=True),
        ATTR_CREATED: created,
        ATTR_SOURCE: str(source),
        ATTR_PROCESSING: str(processing),
        ATTR_PERIOD: str(period),
    }
    # Preserve any pre-existing attrs (e.g. CF Conventions / title) the
    # generator already set; the provenance keys take precedence.
    dataset.attrs.update(attrs)
    return dataset


def read_stamp(dataset: "xarray.Dataset") -> dict:  # noqa: F821
    """Read the provenance stamp back off a dataset's global attributes.

    Returns a dict with the §5a keys; ``generator_config`` is JSON-decoded
    back to a dict. Missing attributes come back as ``None``.
    """
    out: dict = {}
    for attr in STAMP_ATTRS:
        out[attr] = dataset.attrs.get(attr)
    cfg = out.get(ATTR_GENERATOR_CONFIG)
    if isinstance(cfg, str):
        try:
            out[ATTR_GENERATOR_CONFIG] = json.loads(cfg)
        except (ValueError, TypeError):
            pass
    return out
