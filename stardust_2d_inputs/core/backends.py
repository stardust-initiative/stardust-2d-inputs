"""backends — storage backends for the immutable data store.

The loader is **backend-agnostic** (standard §4): identical code runs public
and private, with the backend chosen by config. Every backend resolves a
``content_hash`` to a local file path, fetching the immutable blob from its
store if it is not already in the local content-addressed cache.

Backends
--------
LocalBackend
    A plain directory of content-addressed blobs (development; the migration
    target for Phase C validation against the existing data).
GCSBackend
    A private Google Cloud Storage bucket (standard §4 / §6). Needs the
    ``[gcs]`` extra (``gcsfs``); imported lazily so ``core`` stays lean.
ZenodoBackend
    The public Zenodo deposit — a versioned record, files fetched over
    HTTPS. No credentials, no extra dependency.

The content-addressed local cache
----------------------------------
Every backend caches fetched blobs under ``<cache_dir>/<content_hash>/<name>``
(the hash is sanitized: ``algo:hex`` -> ``algo-hex``). Because blobs are
content-addressed and immutable, a cache hit is unconditionally valid; the
fetched file's hash is verified against ``content_hash`` on download.
"""

from __future__ import annotations

import abc
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import pooch

# Default hash algorithm for content addressing when a bare hex digest is
# given without an explicit ``algo:`` prefix.
DEFAULT_HASH_ALGO = "sha256"


class BackendError(Exception):
    """Raised when a backend cannot fetch or verify a blob."""


# --------------------------------------------------------------------------
# hashing / cache helpers
# --------------------------------------------------------------------------
def _split_hash(content_hash: str) -> "tuple[str, str]":
    """Split ``algo:hexdigest`` into ``(algo, hexdigest)``.

    A bare digest (no colon) is assumed to be ``DEFAULT_HASH_ALGO``.
    """
    if ":" in content_hash:
        algo, _, hexdigest = content_hash.partition(":")
        return algo.lower(), hexdigest.lower()
    return DEFAULT_HASH_ALGO, content_hash.lower()


def _cache_subdir(content_hash: str) -> str:
    """Filesystem-safe per-blob cache subdirectory name."""
    return content_hash.replace(":", "-").replace("/", "_")


def hash_file(path: "os.PathLike | str", algo: str = DEFAULT_HASH_ALGO) -> str:
    """Return the ``algo:hexdigest`` content hash of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def verify_file(path: "os.PathLike | str", content_hash: str) -> bool:
    """True iff ``path`` hashes to ``content_hash``."""
    algo, expected = _split_hash(content_hash)
    actual = hash_file(path, algo).split(":", 1)[1]
    return actual == expected


# --------------------------------------------------------------------------
# the Backend ABC
# --------------------------------------------------------------------------
class Backend(abc.ABC):
    """Abstract storage backend.

    A backend owns a *store* (a directory, a bucket, a Zenodo record) and a
    *local content-addressed cache*. ``fetch()`` is the single public method:
    it returns a local path for a ``(content_hash, filename)`` pair, copying
    the blob into the cache from the store on a cache miss.
    """

    def __init__(self, cache_dir: "Optional[os.PathLike | str]" = None):
        if cache_dir is None:
            cache_dir = pooch.os_cache("stardust_2d_inputs")
        self.cache_dir = Path(cache_dir)

    # ----------------------------------------------------------- interface
    @abc.abstractmethod
    def _retrieve(self, content_hash: str, filename: str, dest: Path) -> None:
        """Fetch the blob ``(content_hash, filename)`` from the store to ``dest``.

        Implementations write the blob to the exact path ``dest`` (its parent
        directory already exists). They need not verify the hash — ``fetch()``
        does that.
        """

    # -------------------------------------------------------------- public
    def cache_path(self, content_hash: str, filename: str) -> Path:
        """Local cache path for a blob (whether or not it is present yet)."""
        return self.cache_dir / _cache_subdir(content_hash) / filename

    def is_cached(self, content_hash: str, filename: str) -> bool:
        """True iff the blob is present in the local cache (hash unverified)."""
        return self.cache_path(content_hash, filename).is_file()

    def fetch(self, content_hash: str, filename: str) -> Path:
        """Return a local path to the blob, fetching + verifying on a miss.

        On a cache hit the cached file is returned directly (content-addressed
        immutable blobs cannot go stale). On a miss the blob is retrieved into
        a temporary file, hash-verified against ``content_hash``, then moved
        into place atomically.
        """
        dest = self.cache_path(content_hash, filename)
        if dest.is_file():
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            self._retrieve(content_hash, filename, tmp)
            if not verify_file(tmp, content_hash):
                raise BackendError(
                    f"hash mismatch for {filename!r}: fetched blob does not "
                    f"match expected {content_hash!r}"
                )
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return dest

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cache_dir={str(self.cache_dir)!r})"


# --------------------------------------------------------------------------
# LocalBackend
# --------------------------------------------------------------------------
class LocalBackend(Backend):
    """A store that is a local directory of content-addressed blobs.

    The store layout mirrors the cache: ``<root>/<content_hash>/<filename>``.
    Used for development and for Phase-C validation against existing data.
    If ``cache_dir`` is not given it defaults to the store root itself — the
    store *is* the cache, so ``fetch()`` becomes a no-op lookup.
    """

    def __init__(
        self,
        root: "os.PathLike | str",
        cache_dir: "Optional[os.PathLike | str]" = None,
    ):
        self.root = Path(root)
        super().__init__(cache_dir if cache_dir is not None else self.root)

    def _store_path(self, content_hash: str, filename: str) -> Path:
        return self.root / _cache_subdir(content_hash) / filename

    def _retrieve(self, content_hash: str, filename: str, dest: Path) -> None:
        src = self._store_path(content_hash, filename)
        if not src.is_file():
            raise BackendError(
                f"blob not found in local store: {src} "
                f"(content_hash={content_hash!r}, filename={filename!r})"
            )
        shutil.copyfile(src, dest)

    def fetch(self, content_hash: str, filename: str) -> Path:
        # When the cache is the store, return the store path directly (after
        # a hash check) — no copy needed.
        if self.cache_dir == self.root:
            src = self._store_path(content_hash, filename)
            if not src.is_file():
                raise BackendError(f"blob not found in local store: {src}")
            if not verify_file(src, content_hash):
                raise BackendError(
                    f"hash mismatch for {filename!r} in local store {src}"
                )
            return src
        return super().fetch(content_hash, filename)


# --------------------------------------------------------------------------
# GCSBackend
# --------------------------------------------------------------------------
class GCSBackend(Backend):
    """A private Google Cloud Storage bucket of content-addressed blobs.

    Blob layout in the bucket: ``gs://<bucket>/<prefix>/<content_hash>/<name>``.
    Authentication is left to the environment — ``gcsfs`` picks up
    Application Default Credentials, or a service-account key file pointed to
    by ``GOOGLE_APPLICATION_CREDENTIALS`` (standard §6: a read-only IAM
    service account).

    Requires the ``[gcs]`` extra (``gcsfs``); the import is lazy so that
    importing ``core`` never requires it.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        cache_dir: "Optional[os.PathLike | str]" = None,
        project: "Optional[str]" = None,
        token: "Optional[str]" = None,
    ):
        super().__init__(cache_dir)
        self.bucket = bucket.replace("gs://", "").strip("/")
        self.prefix = prefix.strip("/")
        self.project = project
        self.token = token
        self._fs = None  # lazily constructed gcsfs.GCSFileSystem

    def _filesystem(self):
        if self._fs is None:
            try:
                import gcsfs  # noqa: F401  (optional [gcs] dependency)
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise BackendError(
                    "GCSBackend requires the 'gcs' extra: "
                    "pip install 'stardust_2d_inputs[gcs]'"
                ) from exc
            self._fs = gcsfs.GCSFileSystem(project=self.project, token=self.token)
        return self._fs

    def _blob_path(self, content_hash: str, filename: str) -> str:
        parts = [self.bucket]
        if self.prefix:
            parts.append(self.prefix)
        parts.append(_cache_subdir(content_hash))
        parts.append(filename)
        return "/".join(parts)

    def _retrieve(self, content_hash: str, filename: str, dest: Path) -> None:
        fs = self._filesystem()
        remote = self._blob_path(content_hash, filename)
        if not fs.exists(remote):
            raise BackendError(
                f"blob not found in GCS bucket: gs://{remote} "
                f"(content_hash={content_hash!r})"
            )
        fs.get_file(remote, str(dest))


# --------------------------------------------------------------------------
# ZenodoBackend
# --------------------------------------------------------------------------
class ZenodoBackend(Backend):
    """The public Zenodo deposit — a versioned record fetched over HTTPS.

    A Zenodo *record* is an immutable, version-DOI'd set of files; the loader
    pins one record id (standard §5c, public side). Files are addressed by
    *filename* within the record:
    ``https://zenodo.org/records/<record_id>/files/<filename>``.

    The ``content_hash`` from the registry is still used as the local-cache
    key and is verified after download. Public Zenodo files carry an MD5
    checksum in the record metadata; when the registry hash is an ``md5:``
    hash it therefore corresponds directly to Zenodo's own checksum.

    No credentials and no extra dependency — Zenodo records are public.
    """

    BASE_URL = "https://zenodo.org"

    def __init__(
        self,
        record_id: "str | int",
        cache_dir: "Optional[os.PathLike | str]" = None,
        base_url: "Optional[str]" = None,
    ):
        super().__init__(cache_dir)
        self.record_id = str(record_id)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")

    def file_url(self, filename: str) -> str:
        """Public download URL of ``filename`` within the pinned record."""
        return f"{self.base_url}/records/{self.record_id}/files/{filename}"

    def _retrieve(self, content_hash: str, filename: str, dest: Path) -> None:
        url = self.file_url(filename)
        try:
            with urllib.request.urlopen(url) as resp:  # noqa: S310 (public URL)
                if getattr(resp, "status", 200) >= 400:
                    raise BackendError(f"Zenodo returned HTTP {resp.status} for {url}")
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
        except BackendError:
            raise
        except Exception as exc:  # pragma: no cover - network-dependent
            raise BackendError(
                f"failed to fetch {filename!r} from Zenodo record "
                f"{self.record_id} ({url}): {exc}"
            ) from exc
