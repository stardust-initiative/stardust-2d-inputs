"""Refine angular coefficients and extend an existing RRTMG optical table.

Non-angular parent rows are retained exactly. Angular rows are all recomputed.
The parent supplies the radius/zenith grids and spectral weighting. Material
refractive indices and band integration use the maintained optical generator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

from . import _common as c, silica, calcite
from ._angular import beta_calc

ANGULAR = ("bet_bar_bsca_sw", "bet_mu_bsca_sw")
FIELDS = ("babs_lw", "bext_sw", "bsca_sw", "basc_sw") + ANGULAR
_CONTEXT = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extended_grid(parent, extra_rows=17):
    """Append log-radius rows without shifting any existing sample."""
    x = np.asarray(parent, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite one-dimensional log-radius grid")
    step = (x[-1] - x[0]) / (len(x) - 1)
    if step <= 0 or not np.allclose(np.diff(x), step, rtol=1e-10, atol=1e-14):
        raise ValueError("Parent radius grid must be uniformly log-spaced")
    if not isinstance(extra_rows, int) or extra_rows < 0:
        raise ValueError("extra_rows must be a nonnegative integer")
    return np.r_[x, x[-1] + step * np.arange(1, extra_rows + 1)]


class RefinedParticle(c.Particle):
    """Use the qualified angular integral without modifying the legacy class."""

    beta_calc = staticmethod(beta_calc)


def _initialize(parent, ri_csv, material, nquad):
    global _CONTEXT
    with xr.open_dataset(parent) as ds:
        old = ds.load()
    sw = np.exp(-np.linspace(np.log(1 / 0.2), np.log(1 / 15.0), 501))
    lw = np.linspace(0.2, 1000.0, 20001)
    _CONTEXT = c._build_radius_context(
        RefinedParticle(material, sw, ri_csv),
        RefinedParticle(material, lw, ri_csv),
        old.coszen.values,
        nquad,
        c.RadiationTransferTableRRTMG.NU_WN_SW1,
        c.RadiationTransferTableRRTMG.NU_WN_SW2,
        c.RRTMG_LW_BAND_BOUNDS.astype(float),
        old["spec sw T"].values,
        old["spec lw T"].values,
        np.array([-np.inf, np.inf]),
        np.r_[old["spec lw band1"].values, old["spec lw band2"].values[-1]],
        sw,
        lw,
        len(old.coszen),
    )


def _row(task):
    i, radius, target = task
    _, rows = c._compute_radius(i, radius, _CONTEXT)
    np.savez(target, **rows)
    return i


def assemble(parent, output, logs, rows, provenance):
    """Validate and assemble rows, retaining parent non-angular values exactly."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with nc.Dataset(parent) as old:
        n = len(old.dimensions["mu_samples"])
        for key in FIELDS:
            if not np.all(np.isfinite(rows[key])):
                raise ValueError(f"Nonfinite coefficients: {key}")
            if key not in ANGULAR:
                np.testing.assert_allclose(
                    old[key][:], rows[key][:n], rtol=1e-10, atol=1e-25
                )
        for key in ("babs_lw", "bsca_sw", "bext_sw") + ANGULAR:
            if np.min(rows[key]) < -1e-25:
                raise ValueError(f"Negative coefficients: {key}")
        sca = rows["bsca_sw"]
        for test in (
            rows["bext_sw"] >= sca - 1e-25,
            np.abs(rows["basc_sw"]) <= sca + 1e-25,
            rows[ANGULAR[0]] <= sca + 1e-25,
            rows[ANGULAR[1]] <= sca[:, :, None] + 1e-25,
        ):
            if not np.all(test):
                raise ValueError("Optical coefficient bounds failed")
        # Validate schema before opening the output so unsupported fields fail cleanly.
        for key, var in old.variables.items():
            if "mu_samples" in var.dimensions and key not in FIELDS + ("mu_samples",):
                raise ValueError(f"Unsupported radius-dependent field: {key}")
        with nc.Dataset(output, "w") as new:
            new.setncatts({a: old.getncattr(a) for a in old.ncattrs()})
            new.setncatts(
                {
                    "refinement_parent_sha256": sha256(parent),
                    "refinement_provenance": json.dumps(provenance, sort_keys=True),
                    "refinement_created_utc": datetime.now(timezone.utc).isoformat(),
                    "refinement_method": "Legendre hemisphere integration; all angular rows recomputed; "
                    "parent non-angular rows retained exactly",
                }
            )
            for key, dim in old.dimensions.items():
                new.createDimension(key, len(logs) if key == "mu_samples" else len(dim))
            for key, var in old.variables.items():
                kw = (
                    {"fill_value": var.getncattr("_FillValue")}
                    if "_FillValue" in var.ncattrs()
                    else {}
                )
                dest = new.createVariable(key, var.datatype, var.dimensions, **kw)
                dest.setncatts(
                    {a: var.getncattr(a) for a in var.ncattrs() if a != "_FillValue"}
                )
                if key == "mu_samples":
                    dest[:] = logs
                elif key in ANGULAR:
                    dest[:] = rows[key]
                elif key in FIELDS:
                    dest[:n] = var[:]
                    dest[n:] = rows[key][n:]
                else:
                    dest[:] = var[:]


def generate(parent, output, material, work_dir, extra_rows=17, workers=1):
    """Compute 512/1024-node rows, require convergence, then write a new table.

    Work directories are deliberately exclusive: stale rows cannot be reused
    across changed generators, input tables or settings.
    """
    parent, output, work_dir = Path(parent), Path(output), Path(work_dir)
    if output.exists():
        raise FileExistsError(output)
    modules = {"silica": silica, "calcite": calcite}
    module = modules[material]
    with xr.open_dataset(parent) as old:
        logs = extended_grid(old.mu_samples.values, extra_rows)
    work_dir.mkdir(parents=True, exist_ok=False)
    ri = work_dir / "refractive_index.csv"
    c.write_refractive_index_csv(str(ri), *module._refractive_index(c.OpticalConfig()))
    result = {}
    for nquad in (512, 1024):
        directory = work_dir / str(nquad)
        directory.mkdir()
        tasks = [
            (i, float(np.exp(r)), str(directory / f"{i:03}.npz"))
            for i, r in enumerate(logs)
        ]
        args = (str(parent), str(ri), material, nquad)
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_initialize, initargs=args
            ) as pool:
                list(pool.map(_row, tasks))
        else:
            _initialize(*args)
            for task in tasks:
                _row(task)
        collected = {k: [] for k in FIELDS}
        for i in range(len(logs)):
            with np.load(directory / f"{i:03}.npz") as data:
                for key in FIELDS:
                    collected[key].append(data[key])
        result[nquad] = {k: np.asarray(v) for k, v in collected.items()}
    for key in FIELDS:
        np.testing.assert_allclose(
            result[512][key], result[1024][key], rtol=1e-9, atol=1e-25
        )
    provenance = {
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "miepython", "pandas", "netCDF4", "xarray")
        },
        "material": material,
        "extra_rows": extra_rows,
        "degree": 384,
        "quadrature_orders": [512, 1024],
        "selected_order": 1024,
        "parent_sha256": sha256(parent),
        "ri_sha256": sha256(ri),
        "generator_sha256": sha256(__file__),
        "angular_sha256": sha256(Path(__file__).with_name("_angular.py")),
        "common_sha256": sha256(c.__file__),
        "convergence_rtol": 1e-9,
        "convergence_atol": 1e-25,
    }
    assemble(parent, output, logs, result[1024], provenance)
    (work_dir / "provenance.json").write_text(
        json.dumps({**provenance, "output_sha256": sha256(output)}, indent=2)
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--material", required=True, choices=["silica", "calcite"])
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--extra-rows", type=int, default=17)
    parser.add_argument("--workers", type=int, default=1)
    generate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
