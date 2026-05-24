"""Shared pytest fixtures: a tiny content-addressed LocalBackend store.

Builds, in a per-session temp directory:

* 3 tiny fixture NetCDF files (a "variables" dataset with two versions, a
  cloud-cover dataset, a probationary transport dataset);
* a content-addressed store directory laid out as the LocalBackend expects;
* a fixture ``registry.json`` whose ``content_hash`` values are the real
  hashes of those files;
* a fixture release manifest pinning a coherent set.

Everything is generated at runtime — no data files are tracked in the repo.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import xarray as xr

from stardust_2d_inputs.core.backends import LocalBackend, hash_file


def _make_dataset(seed: int, varname: str) -> xr.Dataset:
    """A tiny deterministic lat/lev dataset."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-90, 90, 6)
    lev = np.array([100.0, 500.0, 1000.0])
    data = rng.random((lev.size, lat.size)).astype("float64")
    return xr.Dataset(
        {varname: (("lev", "lat"), data)},
        coords={"lat": lat, "lev": lev},
        attrs={"title": f"fixture dataset ({varname}, seed={seed})"},
    )


@pytest.fixture(scope="session")
def data_env(tmp_path_factory) -> dict:
    """Build the fixture store, registry and release manifest.

    Returns a dict with ``root`` (repo root with registry/ + releases/),
    ``store`` (the content-addressed blob store) and ``cache`` (a cache dir).
    """
    base = tmp_path_factory.mktemp("inputs_engine")
    store = base / "store"
    store.mkdir()
    (base / "registry").mkdir()
    (base / "releases").mkdir()

    # --- generate the 3 fixture NetCDF files ------------------------------
    raw_dir = base / "raw"
    raw_dir.mkdir()
    files = {
        "variables_v1": ("Monthly_Zonal_Variables_v1.nc", _make_dataset(1, "T")),
        "variables_v2": ("Monthly_Zonal_Variables_v2.nc", _make_dataset(2, "T")),
        "cloud_cover": ("Monthly_Zonal_Cloud_Cover.nc", _make_dataset(3, "cc")),
        "tem": ("diffusion_ml_TEMdecomp_raw.nc", _make_dataset(4, "Kpp")),
    }
    meta = {}
    for fid, (fname, ds) in files.items():
        raw_path = raw_dir / fname
        ds.to_netcdf(raw_path)
        chash = hash_file(raw_path)  # sha256:...
        # place into the content-addressed store: <store>/<algo-hex>/<fname>
        blob_dir = store / chash.replace(":", "-")
        blob_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_path, blob_dir / fname)
        meta[fid] = {"filename": fname, "content_hash": chash}

    # --- fixture registry.json -------------------------------------------
    registry = {
        "schema_version": 1,
        "datasets": {
            "Monthly_Zonal_Variables": {
                "description": "fixture variables dataset (two versions)",
                "versions": [
                    {
                        "version": "v1",
                        "content_hash": meta["variables_v1"]["content_hash"],
                        "filename": meta["variables_v1"]["filename"],
                        "status": "verified",
                        "period": "2008-2017",
                    },
                    {
                        "version": "v2",
                        "content_hash": meta["variables_v2"]["content_hash"],
                        "filename": meta["variables_v2"]["filename"],
                        "status": "verified",
                        "period": "2008-2018",
                    },
                ],
            },
            "Monthly_Zonal_Cloud_Cover": {
                "description": "fixture cloud-cover dataset",
                "versions": [
                    {
                        "version": "v1",
                        "content_hash": meta["cloud_cover"]["content_hash"],
                        "filename": meta["cloud_cover"]["filename"],
                        "status": "verified",
                    }
                ],
            },
            "diffusion_ml_TEMdecomp_raw": {
                "description": "fixture transport dataset, on probation",
                "versions": [
                    {
                        "version": "v1",
                        "content_hash": meta["tem"]["content_hash"],
                        "filename": meta["tem"]["filename"],
                        "status": "probation",
                    }
                ],
            },
        },
    }
    registry_path = base / "registry" / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # --- fixture release manifest ----------------------------------------
    # Pins Variables to the OLDER v1 even though v2 is also verified -> lets
    # the resolve-by-release test prove the pin overrides "latest verified".
    release = {
        "release": "fixture-release",
        "description": "fixture release manifest",
        "created": "2026-05-19",
        "pins": {
            "Monthly_Zonal_Variables": "v1",
            "Monthly_Zonal_Cloud_Cover": "v1",
        },
    }
    (base / "releases" / "fixture-release.json").write_text(
        json.dumps(release, indent=2), encoding="utf-8"
    )

    return {
        "root": base,
        "store": store,
        "cache": base / "cache",
        "registry_path": registry_path,
        "releases_dir": base / "releases",
        "meta": meta,
        "files": {fid: (raw_dir / fname) for fid, (fname, _) in files.items()},
    }


@pytest.fixture
def local_backend(data_env) -> LocalBackend:
    """A LocalBackend over the fixture store, with a separate cache dir."""
    return LocalBackend(root=data_env["store"], cache_dir=data_env["cache"])
