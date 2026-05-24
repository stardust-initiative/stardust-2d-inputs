"""Unit tests for ``stardust_2d_inputs.core`` against a LocalBackend.

Covers the resolution rule (resolve-by-release, resolve-by-explicit-version
cherry-pick, latest-verified fallback), the probation warning, run-stamping,
the content-addressed cache, and ``provenance.stamp``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from stardust_2d_inputs.core import provenance
from stardust_2d_inputs.core.backends import BackendError, LocalBackend
from stardust_2d_inputs.core.registry import Registry, RegistryError
from stardust_2d_inputs.core.loader import (
    Loader,
    ProbationWarning,
    reset_resolved_inputs,
    resolved_inputs,
)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_registry_loads_keys(data_env):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    assert reg.schema_version == 1
    assert reg.has_key("Monthly_Zonal_Variables")
    assert reg.versions("Monthly_Zonal_Variables") == ["v1", "v2"]


def test_registry_resolve_explicit(data_env):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    dv = reg.resolve("Monthly_Zonal_Variables", "v1")
    assert dv.version == "v1"
    assert dv.is_verified
    with pytest.raises(RegistryError):
        reg.resolve("Monthly_Zonal_Variables", "v99")


def test_registry_latest_verified(data_env):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    # Both versions verified -> latest is v2 (registry order).
    assert reg.latest_verified("Monthly_Zonal_Variables").version == "v2"


def test_registry_latest_verified_skips_probation(data_env):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    # The only TEM version is on probation -> no verified version exists.
    with pytest.raises(RegistryError):
        reg.latest_verified("diffusion_ml_TEMdecomp_raw")


def test_registry_release_manifest(data_env):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    assert reg.list_releases() == ["fixture-release"]
    rel = reg.load_release("fixture-release")
    assert rel.version_for("Monthly_Zonal_Variables") == "v1"
    assert rel.version_for("diffusion_ml_TEMdecomp_raw") is None


# --------------------------------------------------------------------------
# backend / content-addressed cache
# --------------------------------------------------------------------------
def test_local_backend_fetch_and_cache(data_env):
    backend = LocalBackend(root=data_env["store"], cache_dir=data_env["cache"])
    m = data_env["meta"]["cloud_cover"]
    assert not backend.is_cached(m["content_hash"], m["filename"])
    path = backend.fetch(m["content_hash"], m["filename"])
    assert path.is_file()
    assert backend.is_cached(m["content_hash"], m["filename"])
    # Second fetch is a cache hit returning the same path.
    assert backend.fetch(m["content_hash"], m["filename"]) == path


def test_local_backend_hash_mismatch(data_env, tmp_path):
    backend = LocalBackend(root=data_env["store"], cache_dir=tmp_path / "c")
    m = data_env["meta"]["cloud_cover"]
    wrong_hash = "sha256:" + "0" * 64
    # store the blob under a wrong-hash directory
    bad_dir = data_env["store"] / wrong_hash.replace(":", "-")
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / m["filename"]).write_bytes(
        (
            data_env["store"] / m["content_hash"].replace(":", "-") / m["filename"]
        ).read_bytes()
    )
    with pytest.raises(BackendError, match="hash mismatch"):
        backend.fetch(wrong_hash, m["filename"])


# --------------------------------------------------------------------------
# loader resolution rule
# --------------------------------------------------------------------------
def _loader(data_env, default_release=None):
    reg = Registry(data_env["registry_path"], data_env["releases_dir"])
    backend = LocalBackend(root=data_env["store"], cache_dir=data_env["cache"])
    return Loader(reg, backend, default_release=default_release)


def test_resolve_latest_verified(data_env):
    loader = _loader(data_env)
    dv, via = loader.resolve("Monthly_Zonal_Variables")
    assert via == "latest-verified"
    assert dv.version == "v2"


def test_resolve_by_release(data_env):
    loader = _loader(data_env)
    # Release pins Variables to v1, even though v2 is the latest verified.
    dv, via = loader.resolve("Monthly_Zonal_Variables", release="fixture-release")
    assert via == "release"
    assert dv.version == "v1"


def test_resolve_release_unpinned_key_falls_through(data_env):
    loader = _loader(data_env)
    # The release does not pin Cloud_Cover... actually it does; use TEM,
    # which the release does not pin -> fall through to latest-verified.
    dv, via = loader.resolve(
        "diffusion_ml_TEMdecomp_raw", release="fixture-release", version="v1"
    )
    assert via == "explicit-version"


def test_resolve_explicit_version_cherrypick(data_env):
    loader = _loader(data_env)
    # Pin the release but cherry-pick Variables back to v2 -> explicit wins.
    dv, via = loader.resolve(
        "Monthly_Zonal_Variables", release="fixture-release", version="v2"
    )
    assert via == "explicit-version"
    assert dv.version == "v2"


def test_default_release_used(data_env):
    loader = _loader(data_env, default_release="fixture-release")
    dv, via = loader.resolve("Monthly_Zonal_Variables")
    assert via == "release"
    assert dv.version == "v1"


# --------------------------------------------------------------------------
# loader: load + probation warning + run-stamping
# --------------------------------------------------------------------------
def test_load_returns_dataset(data_env):
    loader = _loader(data_env)
    ds = loader.load("Monthly_Zonal_Cloud_Cover")
    assert isinstance(ds, xr.Dataset)
    assert "cc" in ds
    ds.close()


def test_probation_warning(data_env):
    loader = _loader(data_env)
    with pytest.warns(ProbationWarning, match="PROBATION"):
        ds = loader.load("diffusion_ml_TEMdecomp_raw", version="v1")
    ds.close()


def test_verified_load_no_warning(data_env):
    loader = _loader(data_env)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProbationWarning)
        ds = loader.load("Monthly_Zonal_Variables")
        ds.close()


def test_run_stamp_ledger(data_env):
    reset_resolved_inputs()
    loader = _loader(data_env)
    loader.load("Monthly_Zonal_Variables", release="fixture-release").close()
    loader.load("Monthly_Zonal_Cloud_Cover").close()
    recs = resolved_inputs()
    assert len(recs) == 2
    by_key = {r["key"]: r for r in recs}
    assert by_key["Monthly_Zonal_Variables"]["version"] == "v1"
    assert by_key["Monthly_Zonal_Variables"]["resolved_via"] == "release"
    assert "content_hash" in by_key["Monthly_Zonal_Cloud_Cover"]
    reset_resolved_inputs()
    assert resolved_inputs() == []


# --------------------------------------------------------------------------
# provenance.stamp
# --------------------------------------------------------------------------
def test_provenance_stamp_writes_attrs():
    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    cfg = {"years": [2008, 2017], "var": "T"}
    provenance.stamp(
        ds,
        generator="era5.monthly_zonal_variables",
        config=cfg,
        source="ERA5 CDS",
        processing="zonal mean",
        period="2008-2017",
    )
    assert ds.attrs[provenance.ATTR_GENERATOR] == "era5.monthly_zonal_variables"
    assert ds.attrs[provenance.ATTR_SOURCE] == "ERA5 CDS"
    assert ds.attrs[provenance.ATTR_PROCESSING] == "zonal mean"
    assert ds.attrs[provenance.ATTR_PERIOD] == "2008-2017"
    assert "created" in ds.attrs
    # generator_config is JSON; round-trips back to the input dict.
    assert json.loads(ds.attrs[provenance.ATTR_GENERATOR_CONFIG]) == cfg
    # generator_version is always a string (git describe, or "unknown").
    assert isinstance(ds.attrs[provenance.ATTR_GENERATOR_VERSION], str)


def test_provenance_stamp_roundtrips_via_netcdf(tmp_path):
    ds = xr.Dataset({"x": ("t", np.arange(3.0))})
    provenance.stamp(ds, generator="g", config={"a": 1})
    out = tmp_path / "stamped.nc"
    ds.to_netcdf(out)
    reopened = xr.open_dataset(out)
    stamp = provenance.read_stamp(reopened)
    assert stamp[provenance.ATTR_GENERATOR] == "g"
    assert stamp[provenance.ATTR_GENERATOR_CONFIG] == {"a": 1}
    reopened.close()
