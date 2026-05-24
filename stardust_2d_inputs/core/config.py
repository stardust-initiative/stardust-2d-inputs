"""config — configuration loading and backend selection.

The engine is configured by a JSON file (``config.json``) which is *never*
tracked — only ``config.example.json`` is committed (standard §3 / §7).
``config.json`` selects the storage backend and points the loader at the
registry.

Config schema (``config.json``)
--------------------------------
::

    {
      "backend": "local" | "gcs" | "zenodo",
      "registry_path": "registry/registry.json",   # optional; default shown
      "releases_dir":  "releases",                  # optional; default shown
      "cache_dir":     null,                        # optional; null -> OS cache
      "default_release": null,                      # optional; default release pin

      "local":  { "root": "/path/to/local/store" },
      "gcs":    { "bucket": "stardust-2d-inputs",
                  "prefix": "",
                  "project": null },
      "zenodo": { "record_id": "20271742" }
    }

Only the block for the selected ``backend`` need be populated. Credentials
are never stored here — the GCS backend relies on ambient IAM credentials
(standard §6); Zenodo is public.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from . import backends as _backends
from .registry import DEFAULT_REGISTRY_PATH, DEFAULT_RELEASES_DIR, Registry

DEFAULT_CONFIG_FILENAME = "config.json"
EXAMPLE_CONFIG_FILENAME = "config.example.json"

VALID_BACKENDS = frozenset({"local", "gcs", "zenodo"})


class ConfigError(Exception):
    """Raised on a missing or malformed configuration."""


class Config:
    """Resolved engine configuration.

    Attributes
    ----------
    raw :
        The parsed JSON dict.
    backend_name :
        The selected backend: ``"local"``, ``"gcs"`` or ``"zenodo"``.
    repo_root :
        Directory the relative paths (registry, releases) resolve against —
        the directory containing ``config.json``.
    """

    def __init__(self, raw: Dict[str, Any], repo_root: str):
        self.raw = raw
        self.repo_root = os.path.abspath(repo_root)

        self.backend_name = raw.get("backend")
        if self.backend_name not in VALID_BACKENDS:
            raise ConfigError(
                f"config 'backend' must be one of {sorted(VALID_BACKENDS)}, "
                f"got {self.backend_name!r}"
            )

        self.registry_path = self._abspath(
            raw.get("registry_path", DEFAULT_REGISTRY_PATH)
        )
        self.releases_dir = self._abspath(raw.get("releases_dir", DEFAULT_RELEASES_DIR))
        cache_dir = raw.get("cache_dir")
        self.cache_dir = self._abspath(cache_dir) if cache_dir else None
        self.default_release: Optional[str] = raw.get("default_release")

    # ------------------------------------------------------------ helpers
    def _abspath(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_root, path)

    # -------------------------------------------------------------- build
    def build_registry(self) -> Registry:
        """Construct the :class:`Registry` described by this config."""
        return Registry(
            registry_path=self.registry_path,
            releases_dir=self.releases_dir,
        )

    def build_backend(self) -> _backends.Backend:
        """Construct the storage :class:`Backend` selected by this config."""
        block = self.raw.get(self.backend_name, {})

        if self.backend_name == "local":
            root = block.get("root")
            if not root:
                raise ConfigError("local backend requires 'local.root'")
            return _backends.LocalBackend(
                root=self._abspath(root), cache_dir=self.cache_dir
            )

        if self.backend_name == "gcs":
            bucket = block.get("bucket")
            if not bucket:
                raise ConfigError("gcs backend requires 'gcs.bucket'")
            return _backends.GCSBackend(
                bucket=bucket,
                prefix=block.get("prefix", ""),
                project=block.get("project"),
                cache_dir=self.cache_dir,
            )

        if self.backend_name == "zenodo":
            record_id = block.get("record_id")
            if not record_id:
                raise ConfigError("zenodo backend requires 'zenodo.record_id'")
            return _backends.ZenodoBackend(
                record_id=record_id,
                base_url=block.get("base_url"),
                cache_dir=self.cache_dir,
            )

        # Unreachable: backend_name validated in __init__.
        raise ConfigError(f"unsupported backend {self.backend_name!r}")

    def __repr__(self) -> str:
        return (
            f"Config(backend={self.backend_name!r}, " f"repo_root={self.repo_root!r})"
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Config:
    """Load a :class:`Config` from a ``config.json`` file.

    Resolution of ``path``:

    1. The explicit ``path`` argument, if given.
    2. The ``STARDUST_2D_INPUTS_CONFIG`` environment variable.
    3. ``config.json`` in the current working directory.

    The repository root (against which relative paths in the config resolve)
    is taken to be the directory containing the config file.
    """
    if path is None:
        path = os.environ.get("STARDUST_2D_INPUTS_CONFIG")
    if path is None:
        path = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)

    if not os.path.isfile(path):
        raise ConfigError(
            f"config file not found: {path}. Copy {EXAMPLE_CONFIG_FILENAME} "
            f"to {DEFAULT_CONFIG_FILENAME} and edit it, or set "
            f"STARDUST_2D_INPUTS_CONFIG."
        )

    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file {path} is not valid JSON: {exc}") from exc

    return Config(raw, repo_root=os.path.dirname(os.path.abspath(path)))


def config_from_dict(raw: Dict[str, Any], repo_root: str) -> Config:
    """Build a :class:`Config` directly from a dict (used by tests)."""
    return Config(raw, repo_root=repo_root)
