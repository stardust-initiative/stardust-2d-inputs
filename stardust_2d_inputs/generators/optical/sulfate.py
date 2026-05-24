"""optical.sulfate — sulfate-aerosol optical-table generator.

Two-stage pipeline for stratospheric sulfate aerosol (75% sulphuric-acid
solution):

  Stage 1 — build the complex refractive index ``m(lambda) = n + i*k`` by
            stitching three published sources onto a common log-spaced
            wavelength grid: a broadband background table is overlaid, where
            available, by the more specific measurements of the other two
            sources.
  Stage 2 — Mie + RRTMG-band averaging into the climlab optical table
            (see :mod:`optical._common`).

References
    ref1  Gosse, Labrie & Chylek, "Refractive index of dry and aqueous
          sulphuric acid: imaginary part of the refractive index of sulfates
          and nitrates in the 0.7-2.6 um spectral region", Applied Optics
          36(16), 3622-3628 (1997).
    ref2  Ferraro, Charlton-Perez & Highwood, "Stratospheric dynamics and
          midlatitude jets under geoengineering with space mirrors and
          sulfate and titania aerosols", Journal of Geophysical Research:
          Atmospheres 120(2), 414-429 (2015), doi:10.1002/2014JD022734
          (75% sulphuric-acid index, SRA data).
    ref3  Hummel, Shettle & Longtin, "A New Background Stratospheric Aerosol
          Model for Use in Atmospheric Radiation Models", AFGL-TR-88-0166
          (1988).

Raw inputs (``optical/data/sulfate/``)
    ``ref3_sulfate_table_h2so4_75percent.csv`` — broadband background table;
    ``sulfate_data_s1.txt`` — ref2 75% sulphuric-acid index;
    ``sulfate_absorption_ref1.csv`` — ref1 absorption (imaginary index) for
    several sulfate / nitrate solutions;
    ``REFERENCES.txt`` — the full literature citations for those files.

Configuration is via :class:`optical._common.OpticalConfig`; the output is
``stardust_particles_H2SO4_75_climlab.nc``, stamped via
:func:`core.provenance.stamp`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import _common
from ._common import OpticalConfig, build_optical_table

GENERATOR = "optical.sulfate"

MATERIAL_NAME = "sulfate"
DEFAULT_OPTICAL_CSV = "./H2SO4_optical.csv"
DEFAULT_OUT_FILE = "./stardust_particles_H2SO4_75_climlab.nc"

SOURCE = (
    "Refractive-index input: 75% H2SO4 complex index stitched from three "
    "published sources. (1) Hummel, Shettle & Longtin, 'A New Background "
    "Stratospheric Aerosol Model for Use in Atmospheric Radiation Models', "
    "AFGL-TR-88-0166 (1988) -- broadband background table. (2) Ferraro, "
    "Charlton-Perez & Highwood, 'Stratospheric dynamics and midlatitude jets "
    "under geoengineering with space mirrors and sulfate and titania "
    "aerosols', Journal of Geophysical Research: Atmospheres 120(2), 414-429 "
    "(2015), doi:10.1002/2014JD022734 -- 75% sulphuric-acid index (SRA "
    "data). (3) Gosse, Labrie & Chylek, 'Refractive index of dry and "
    "aqueous sulphuric acid: imaginary part of the refractive index of "
    "sulfates and nitrates in the 0.7-2.6 um spectral region', Applied "
    "Optics 36(16), 3622-3628 (1997) -- absorption (imaginary index) "
    "measurements. Raw input files carried in optical/data/sulfate/."
)
PROCESSING = (
    "Stage 1: broadband background refractive-index table overlaid, where "
    "available, by the Ferraro (2015) and Gosse (1997) measurements. "
    "Stage 2: Mie efficiencies (miepython) Planck-weighted into the RRTMG "
    "shortwave/longwave band tables."
)

_SULFATE_DATA_DIR = os.path.join(_common.DATA_DIR, "sulfate")

# Imaginary-index column selected from the ref1 multi-solution absorption
# table (75% sulphuric acid).
_REF1_K_LABEL = "H2SO4 72%"


def _load_ref1():
    """ref1 (Gosse 1997): imaginary index for 75% H2SO4 vs wavelength.

    The real part is unknown in this source (returned as NaN, so the Stage-1
    overlay leaves the background real part in place).
    """
    path = os.path.join(_SULFATE_DATA_DIR, "sulfate_absorption_ref1.csv")
    df = pd.read_csv(path)
    df["Wavelength (um)"] = 1e4 / df["Wavenumber [cm^-1]"]
    df = df.dropna(subset=[_REF1_K_LABEL])
    vlambda = np.array(df["Wavelength (um)"])[::-1]
    n_vect = np.full(len(df), np.nan)
    k_vect = np.array(df[_REF1_K_LABEL])[::-1]
    return vlambda, n_vect, k_vect


def _load_ref2():
    """ref2 (Ferraro 2014): real + imaginary index for 75% sulphuric acid."""
    path = os.path.join(_SULFATE_DATA_DIR, "sulfate_data_s1.txt")
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    # First four lines are headers; data rows start at index 4.
    data = np.zeros((0, 3))
    for ln in lines[4:]:
        if "*" in ln:
            continue
        row = np.array([float(s) for s in ln.strip().split("     ")])
        data = np.vstack([data, row])
    df = pd.DataFrame(
        data=data, columns=["Wavelength (m)", "Real Part", "Imaginary Part"]
    )
    df["Wavelength (um)"] = 1e6 * df["Wavelength (m)"]
    return (
        np.array(df["Wavelength (um)"]),
        np.array(df["Real Part"]),
        np.array(df["Imaginary Part"]),
    )


def _load_ref3():
    """ref3 (Hummel 1988): broadband background 75% H2SO4 refractive index."""
    path = os.path.join(_SULFATE_DATA_DIR, "ref3_sulfate_table_h2so4_75percent.csv")
    df = pd.read_csv(path)
    return (np.array(df["LAMBDA(um)"]), np.array(df["nr"]), np.array(df["ni"]))


def _overlay(vlambda_vect, n_vect, k_vect, src_lambda, src_n, src_k):
    """Overlay a source spectrum onto the running n / k arrays.

    Within the source's wavelength span, replace ``n`` / ``k`` with the
    interpolated source values wherever those are not NaN.
    """
    v1, v2 = src_lambda.min(), src_lambda.max()
    ind = np.where(np.abs(vlambda_vect - 0.5 * (v1 + v2)) < 0.5 * (v2 - v1))[0]
    n_tmp = np.interp(vlambda_vect[ind], src_lambda, src_n)
    k_tmp = np.interp(vlambda_vect[ind], src_lambda, src_k)
    for i, idx in enumerate(ind):
        if not np.isnan(n_tmp[i]):
            n_vect[idx] = n_tmp[i]
        if not np.isnan(k_tmp[i]):
            k_vect[idx] = k_tmp[i]
    return n_vect, k_vect


def _refractive_index(cfg: OpticalConfig):
    """Stage 1: build sulfate ``(vlambda_vect, n_vect, k_vect)``.

    Start from the ref3 broadband background, then overlay ref2 and ref1
    where each is defined.
    """
    vlambda_vect = np.exp(
        np.linspace(
            np.log(cfg.ri_lambda_min), np.log(cfg.ri_lambda_max), cfg.ri_n_lambda
        )
    )

    lam1, n1, k1 = _load_ref1()
    lam2, n2, k2 = _load_ref2()
    lam3, n3, k3 = _load_ref3()

    # Background: ref3 interpolated onto the full grid.
    n_vect = np.interp(vlambda_vect, lam3, n3)
    k_vect = np.interp(vlambda_vect, lam3, k3)

    # Overlay the more specific measurements.
    n_vect, k_vect = _overlay(vlambda_vect, n_vect, k_vect, lam2, n2, k2)
    n_vect, k_vect = _overlay(vlambda_vect, n_vect, k_vect, lam1, n1, k1)

    return vlambda_vect, n_vect, k_vect


def generate(
    optical_csv: str = DEFAULT_OPTICAL_CSV,
    out_file: str = DEFAULT_OUT_FILE,
    cfg: OpticalConfig | None = None,
) -> str:
    """Build the sulfate climlab optical table and stamp it.

    Writes the intermediate refractive-index CSV to ``optical_csv`` and the
    climlab RRTMG table to ``out_file``. Returns the output NetCDF path.
    """
    cfg = cfg or OpticalConfig()
    cfg.optical_csv = optical_csv
    cfg.out_file = out_file
    return build_optical_table(
        MATERIAL_NAME,
        cfg,
        _refractive_index,
        generator=GENERATOR,
        source=SOURCE,
        processing=PROCESSING,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--optical-csv",
        default=DEFAULT_OPTICAL_CSV,
        help="intermediate refractive-index CSV "
        "(Stage-1 output; default: %(default)s)",
    )
    parser.add_argument(
        "--out-file",
        default=DEFAULT_OUT_FILE,
        help="output climlab NetCDF path (default: %(default)s)",
    )
    parser.add_argument(
        "--nr",
        type=int,
        default=OpticalConfig.nr,
        help="number of Mie radius samples (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=OpticalConfig.n_workers,
        help="worker processes for the radius loop "
        "(<=1 serial; default: %(default)s)",
    )
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = OpticalConfig(nr=args.nr, n_workers=args.workers)
    generate(args.optical_csv, args.out_file, cfg)


if __name__ == "__main__":
    main()
