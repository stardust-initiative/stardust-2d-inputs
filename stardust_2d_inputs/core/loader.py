"""loader — the public ``load(key, ...)`` entry point.

``load`` resolves a dataset key to an immutable blob, fetches it through the
configured backend, opens it as an :class:`xarray.Dataset`, and records the
resolved ``(key, version)`` for run-stamping.

Resolution rule
---------------
For a ``load(key, release=..., version=...)`` call, the version is resolved
in this strict order of precedence:

1. **Explicit version** — if ``version`` is given, that exact registry entry
   is used. This is the cherry-pick / bisection path (standard §5c): pin a
   release but override one key to an older hash.
2. **Release pin** — else, if a ``release`` is given (or a
   ``default_release`` is set in config) and that release manifest pins the
   key, the release's pinned version is used.
3. **Latest verified** — else, the latest ``verified`` version of the key
   (registry order). A key with no verified version raises rather than
   silently falling back to a ``probation`` row.

A resolved entry whose ``status`` is ``probation`` emits a warning (the data
loads, but the caller is told it is not verified). A ``deprecated`` entry
loads only when reached via path 1 (explicit pin) or path 2 (release pin);
it is never selected by path 3.

Run-stamping
------------
Every resolved ``(key, version, content_hash, status)`` is appended to a
module-level ledger. A simulation run reads :func:`resolved_inputs` at the
end and stamps its own output with the exact inputs it consumed (standard
§5c). :func:`reset_resolved_inputs` clears the ledger (e.g. per run).
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import xarray as xr

from .backends import Backend
from .config import Config, load_config
from .registry import (
    STATUS_DEPRECATED,
    STATUS_PROBATION,
    DatasetVersion,
    Registry,
)


class ProbationWarning(UserWarning):
    """Emitted when a loaded dataset's registry status is ``probation``."""


# --------------------------------------------------------------------------
# the run-stamp ledger
# --------------------------------------------------------------------------
_resolved_lock = threading.Lock()
_resolved_inputs: List[Dict[str, str]] = []


def _record_resolution(dv: DatasetVersion, resolved_via: str) -> None:
    with _resolved_lock:
        _resolved_inputs.append(
            {
                "key": dv.key,
                "version": dv.version,
                "content_hash": dv.content_hash,
                "status": dv.status,
                "resolved_via": resolved_via,
            }
        )


def resolved_inputs() -> List[Dict[str, str]]:
    """Return the list of ``(key, version, ...)`` records resolved so far.

    A simulation run uses this to stamp its output with its exact inputs.
    A copy is returned; mutating it does not affect the ledger.
    """
    with _resolved_lock:
        return [dict(rec) for rec in _resolved_inputs]


def reset_resolved_inputs() -> None:
    """Clear the run-stamp ledger (call once per run)."""
    with _resolved_lock:
        _resolved_inputs.clear()


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------
class Loader:
    """A configured loader: a :class:`Registry` + a :class:`Backend`.

    The module-level :func:`load` builds a default ``Loader`` from
    ``config.json`` on first use. Construct one explicitly (e.g. in tests)
    to load against a chosen backend / registry without a config file.
    """

    def __init__(
        self,
        registry: Registry,
        backend: Backend,
        default_release: Optional[str] = None,
    ):
        self.registry = registry
        self.backend = backend
        self.default_release = default_release
        self._release_cache: Dict[str, "object"] = {}

    @classmethod
    def from_config(cls, config: Config) -> "Loader":
        return cls(
            registry=config.build_registry(),
            backend=config.build_backend(),
            default_release=config.default_release,
        )

    # ----------------------------------------------------------- resolve
    def resolve(
        self,
        key: str,
        *,
        release: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "tuple[DatasetVersion, str]":
        """Resolve ``key`` to a :class:`DatasetVersion` per the resolution rule.

        Returns ``(dataset_version, resolved_via)`` where ``resolved_via`` is
        one of ``"explicit-version"``, ``"release"`` or ``"latest-verified"``.
        """
        # 1. explicit version — the cherry-pick path.
        if version is not None:
            return self.registry.resolve(key, version), "explicit-version"

        # 2. release pin.
        release_name = release if release is not None else self.default_release
        if release_name is not None:
            rel = self._get_release(release_name)
            pinned = rel.version_for(key)
            if pinned is not None:
                return self.registry.resolve(key, pinned), "release"
            # Release does not pin this key — fall through to latest-verified.

        # 3. latest verified.
        return self.registry.latest_verified(key), "latest-verified"

    def _get_release(self, name: str):
        if name not in self._release_cache:
            self._release_cache[name] = self.registry.load_release(name)
        return self._release_cache[name]

    # -------------------------------------------------------------- fetch
    def fetch_path(
        self,
        key: str,
        *,
        release: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Path:
        """Resolve + fetch ``key``; return a local file path (no open)."""
        dv, resolved_via = self.resolve(key, release=release, version=version)
        self._warn_if_probation(dv)
        path = self.backend.fetch(dv.content_hash, dv.filename)
        _record_resolution(dv, resolved_via)
        return path

    def load(
        self,
        key: str,
        *,
        release: Optional[str] = None,
        version: Optional[str] = None,
        **open_kwargs,
    ) -> xr.Dataset:
        """Resolve, fetch and open ``key`` as an :class:`xarray.Dataset`."""
        dv, resolved_via = self.resolve(key, release=release, version=version)
        self._warn_if_probation(dv)
        path = self.backend.fetch(dv.content_hash, dv.filename)
        ds = xr.open_dataset(path, **open_kwargs)
        _record_resolution(dv, resolved_via)
        return ds

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _warn_if_probation(dv: DatasetVersion) -> None:
        if dv.status == STATUS_PROBATION:
            warnings.warn(
                f"dataset {dv.key!r} version {dv.version!r} is on PROBATION "
                f"(not verified — provenance unconfirmed); using it anyway",
                ProbationWarning,
                stacklevel=3,
            )
        elif dv.status == STATUS_DEPRECATED:
            warnings.warn(
                f"dataset {dv.key!r} version {dv.version!r} is DEPRECATED; "
                f"it was resolved only because it was pinned explicitly",
                ProbationWarning,
                stacklevel=3,
            )


# --------------------------------------------------------------------------
# module-level convenience API
# --------------------------------------------------------------------------
_default_loader: Optional[Loader] = None
_default_loader_lock = threading.Lock()


def get_default_loader() -> Loader:
    """Return the process-wide default loader, building it from config once."""
    global _default_loader
    with _default_loader_lock:
        if _default_loader is None:
            _default_loader = Loader.from_config(load_config())
        return _default_loader


def set_default_loader(loader: Optional[Loader]) -> None:
    """Override (or, with ``None``, reset) the process-wide default loader."""
    global _default_loader
    with _default_loader_lock:
        _default_loader = loader


def load(
    key: str,
    *,
    release: Optional[str] = None,
    version: Optional[str] = None,
    **open_kwargs,
) -> xr.Dataset:
    """Load a dataset by ``key`` from the configured input-data store.

    Parameters
    ----------
    key :
        The dataset key (a registry key, e.g. ``"Monthly_Zonal_Variables"``).
    release :
        Name of a release manifest to pin against. Overrides the config's
        ``default_release``. See the module docstring for the resolution rule.
    version :
        An explicit registry version id. Highest precedence — the cherry-pick
        path for bug bisection.
    **open_kwargs :
        Passed through to :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
    """
    return get_default_loader().load(
        key, release=release, version=version, **open_kwargs
    )
