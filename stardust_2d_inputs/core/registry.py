"""registry — the provenance registry and the release manifests.

The **registry** (``registry/registry.json``) is the git-versioned source of
truth for every dataset the engine knows about (standard §5b / §5c). One
*key* (e.g. ``Monthly_Zonal_Variables``) may have several *versions*; each
version is one immutable, content-addressed blob in a store.

Registry schema (``registry/registry.json``)
---------------------------------------------
::

    {
      "schema_version": 1,
      "datasets": {
        "<key>": {
          "description": "<free text>",
          "versions": [
            {
              "version":      "<version id, e.g. v1>",
              "content_hash": "<algo:hexdigest>",
              "filename":     "<blob filename in the store>",
              "status":       "verified" | "probation" | "deprecated",
              "size_bytes":   <int>,                 # optional
              "source":       "<free text>",         # optional
              "processing":   "<free text>",         # optional
              "period":       "<free text>",         # optional
              "generator":    "<generator name>",    # optional
              "generator_version": "<git describe>", # optional
              "added":        "<ISO-8601 date>"      # optional
            }
          ]
        }
      }
    }

* ``content_hash`` is the address of the immutable blob in the store and the
  key of the local content-addressed cache.
* ``status`` follows §5b: only ``verified`` rows are public-eligible;
  ``probation`` rows resolve but trigger a runtime warning; ``deprecated``
  rows resolve only when pinned explicitly.

Release manifests (``releases/<name>.json``)
--------------------------------------------
A release manifest pins a coherent ``key -> version`` set — "the input set
this paper / this database version was run against". Schema::

    {
      "release":     "<name, e.g. transport-paper-v1>",
      "description": "<free text>",
      "created":     "<ISO-8601 date>",
      "registry_revision": "<git commit/tag of the registry>",  # optional
      "pins": { "<key>": "<version>", ... }
    }

The loader's resolution rule (see ``loader.load``) consults the registry and
(optionally) one release manifest.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# Recognised dataset statuses (standard §5b).
STATUS_VERIFIED = "verified"
STATUS_PROBATION = "probation"
STATUS_DEPRECATED = "deprecated"
VALID_STATUSES = frozenset({STATUS_VERIFIED, STATUS_PROBATION, STATUS_DEPRECATED})

# Default on-disk locations, relative to the repository root.
DEFAULT_REGISTRY_PATH = "registry/registry.json"
DEFAULT_RELEASES_DIR = "releases"


class RegistryError(Exception):
    """Raised when the registry cannot satisfy a resolution request."""


class DatasetVersion:
    """One immutable version of one dataset key."""

    def __init__(self, key: str, entry: dict):
        self.key = key
        self.version: str = entry["version"]
        self.content_hash: str = entry["content_hash"]
        self.filename: str = entry["filename"]
        self.status: str = entry.get("status", STATUS_PROBATION)
        if self.status not in VALID_STATUSES:
            raise RegistryError(
                f"dataset {key!r} version {self.version!r}: unknown status "
                f"{self.status!r} (expected one of {sorted(VALID_STATUSES)})"
            )
        # Optional provenance fields — carried through verbatim.
        self.size_bytes: Optional[int] = entry.get("size_bytes")
        self.source: Optional[str] = entry.get("source")
        self.processing: Optional[str] = entry.get("processing")
        self.period: Optional[str] = entry.get("period")
        self.generator: Optional[str] = entry.get("generator")
        self.generator_version: Optional[str] = entry.get("generator_version")
        self.added: Optional[str] = entry.get("added")
        self._raw = entry

    @property
    def is_verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    @property
    def is_probation(self) -> bool:
        return self.status == STATUS_PROBATION

    @property
    def is_deprecated(self) -> bool:
        return self.status == STATUS_DEPRECATED

    def __repr__(self) -> str:
        return (
            f"DatasetVersion(key={self.key!r}, version={self.version!r}, "
            f"status={self.status!r})"
        )


class Release:
    """A release manifest — a pinned ``key -> version`` set."""

    def __init__(self, manifest: dict):
        self.name: str = manifest["release"]
        self.description: str = manifest.get("description", "")
        self.created: Optional[str] = manifest.get("created")
        self.registry_revision: Optional[str] = manifest.get("registry_revision")
        self.pins: Dict[str, str] = dict(manifest.get("pins", {}))
        self._raw = manifest

    def version_for(self, key: str) -> Optional[str]:
        """Return the version this release pins for ``key``, or ``None``."""
        return self.pins.get(key)

    def __repr__(self) -> str:
        return f"Release(name={self.name!r}, n_pins={len(self.pins)})"


class Registry:
    """In-memory view of ``registry/registry.json`` plus the release manifests.

    Parameters
    ----------
    registry_path :
        Path to ``registry.json``.
    releases_dir :
        Directory holding the release manifests. Defaults to a ``releases/``
        sibling of the registry file. Missing directory is tolerated (no
        releases).
    """

    def __init__(
        self,
        registry_path: str,
        releases_dir: Optional[str] = None,
    ):
        self.registry_path = os.path.abspath(registry_path)
        if releases_dir is None:
            releases_dir = os.path.join(
                os.path.dirname(self.registry_path), os.pardir, DEFAULT_RELEASES_DIR
            )
        self.releases_dir = os.path.abspath(releases_dir)

        self._datasets: Dict[str, Dict[str, DatasetVersion]] = {}
        self._dataset_meta: Dict[str, dict] = {}
        self._version_order: Dict[str, List[str]] = {}
        self.schema_version: Optional[int] = None
        self._load_registry()

    # ----------------------------------------------------------------- load
    @classmethod
    def from_repo_root(cls, repo_root: str) -> "Registry":
        """Build a Registry from a repository root directory."""
        return cls(
            registry_path=os.path.join(repo_root, DEFAULT_REGISTRY_PATH),
            releases_dir=os.path.join(repo_root, DEFAULT_RELEASES_DIR),
        )

    def _load_registry(self) -> None:
        if not os.path.isfile(self.registry_path):
            raise RegistryError(f"registry file not found: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.schema_version = data.get("schema_version")
        datasets = data.get("datasets", {})
        if not isinstance(datasets, dict):
            raise RegistryError("registry 'datasets' must be a JSON object")

        for key, ds in datasets.items():
            self._dataset_meta[key] = {"description": ds.get("description", "")}
            versions = ds.get("versions", [])
            if not versions:
                raise RegistryError(f"dataset {key!r} has no versions")
            by_version: Dict[str, DatasetVersion] = {}
            order: List[str] = []
            for entry in versions:
                dv = DatasetVersion(key, entry)
                if dv.version in by_version:
                    raise RegistryError(
                        f"dataset {key!r}: duplicate version {dv.version!r}"
                    )
                by_version[dv.version] = dv
                order.append(dv.version)
            self._datasets[key] = by_version
            # Registry order is authoritative: later entries are newer.
            self._version_order[key] = order

    # ------------------------------------------------------------- queries
    def keys(self) -> List[str]:
        """All dataset keys known to the registry."""
        return sorted(self._datasets)

    def has_key(self, key: str) -> bool:
        return key in self._datasets

    def describe(self, key: str) -> str:
        """Free-text description of a dataset key."""
        self._require_key(key)
        return self._dataset_meta[key].get("description", "")

    def versions(self, key: str) -> List[str]:
        """All version ids for ``key``, oldest-first (registry order)."""
        self._require_key(key)
        return list(self._version_order[key])

    def resolve(self, key: str, version: str) -> DatasetVersion:
        """Resolve an explicit ``(key, version)`` to its registry entry.

        This is the cherry-pick primitive (standard §5c): the loader uses it
        both for per-key version overrides and as the final step of every
        resolution path.
        """
        self._require_key(key)
        by_version = self._datasets[key]
        if version not in by_version:
            raise RegistryError(
                f"dataset {key!r} has no version {version!r}; "
                f"known versions: {self.versions(key)}"
            )
        return by_version[version]

    def latest_verified(self, key: str) -> DatasetVersion:
        """Return the latest ``verified`` version of ``key``.

        "Latest" = last in registry order (the registry file is append-style;
        newer versions are appended). Raises if the key has no verified
        version — the loader does not silently fall back to a probation row.
        """
        self._require_key(key)
        for version in reversed(self._version_order[key]):
            dv = self._datasets[key][version]
            if dv.is_verified:
                return dv
        raise RegistryError(
            f"dataset {key!r} has no 'verified' version "
            f"(versions: {self.versions(key)}); "
            f"pin an explicit version or a release to use it"
        )

    # ------------------------------------------------------------ releases
    def release_path(self, name: str) -> str:
        """On-disk path of the release manifest ``name`` (``.json`` optional)."""
        fname = name if name.endswith(".json") else f"{name}.json"
        return os.path.join(self.releases_dir, fname)

    def list_releases(self) -> List[str]:
        """Names of the release manifests in ``releases/`` (sans extension)."""
        if not os.path.isdir(self.releases_dir):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.releases_dir)
            if f.endswith(".json")
        )

    def load_release(self, name: str) -> Release:
        """Load a release manifest by name from ``releases/``."""
        path = self.release_path(name)
        if not os.path.isfile(path):
            available = self.list_releases()
            raise RegistryError(
                f"release manifest {name!r} not found at {path}; "
                f"available releases: {available}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        return Release(manifest)

    # ------------------------------------------------------------ internal
    def _require_key(self, key: str) -> None:
        if key not in self._datasets:
            raise RegistryError(
                f"unknown dataset key {key!r}; known keys: {self.keys()}"
            )
