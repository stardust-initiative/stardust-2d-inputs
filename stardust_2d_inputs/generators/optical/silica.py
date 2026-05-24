"""optical.silica — silica optical-table generator.

Two-stage pipeline for amorphous silica (SiO2 glass):

  Stage 1 — build the complex refractive index ``m(lambda) = n + i*k``. The
            shortwave / mid-infrared part is the Kitamura et al. (2007)
            multi-oscillator Lorentzian model (its absorption oscillators plus
            the Kramers-Kronig-consistent real part via the Dawson function).
            Below 8 um the model is replaced by directly sampled ``n`` and
            ``k`` measurements (the carried CSVs).
  Stage 2 — Mie + RRTMG-band averaging into the climlab optical table
            (see :mod:`optical._common`).

Reference
    Kitamura, Pilon & Jonasz, "Optical constants of silica glass from extreme
    ultraviolet to far infrared at near room temperature", Applied Optics
    46(33), 8118-8133 (2007), doi:10.1364/AO.46.008118 (Table 2).

Raw inputs (``optical/data/silica/``)
    ``n_0-1um.csv``, ``n_1-15um.csv``, ``k_0-1um.csv``, ``k_1-15um.csv`` —
    sampled real / imaginary refractive index in the 0-15 um range;
    ``REFERENCES.txt`` — the full literature citation for those files.

Configuration is via :class:`optical._common.OpticalConfig`; the output is
``stardust_particles_silica_climlab.nc``, stamped via
:func:`core.provenance.stamp`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.special import dawsn

from . import _common
from ._common import OpticalConfig, build_optical_table

GENERATOR = "optical.silica"

MATERIAL_NAME = "silica"
DEFAULT_OPTICAL_CSV = "./silica_optical.csv"
DEFAULT_OUT_FILE = "./stardust_particles_silica_climlab.nc"

SOURCE = (
    "Refractive-index input: Kitamura, Pilon & Jonasz, 'Optical constants of "
    "silica glass from extreme ultraviolet to far infrared at near room "
    "temperature', Applied Optics 46(33), 8118-8133 (2007), "
    "doi:10.1364/AO.46.008118. Used as the multi-oscillator Lorentzian model "
    "(Table 2 parameters) plus the sampled n/k measurements carried in "
    "optical/data/silica/ for wavelengths below 8 um."
)
PROCESSING = (
    "Stage 1: Kitamura multi-oscillator refractive-index model with "
    "Kramers-Kronig-consistent real part, replaced below 8 um by sampled "
    "n/k measurements. Stage 2: Mie efficiencies (miepython) Planck-weighted "
    "into the RRTMG shortwave/longwave band tables."
)

_SILICA_DATA_DIR = os.path.join(_common.DATA_DIR, "silica")

# Lambda below which the analytic model is replaced by sampled measurements.
_LAMBDA_ANALYTIC_MIN_UM = 8.0


# ── Kitamura et al. (2007), Table 2 oscillator parameters ──
def _kitamura_params():
    eps_inf = 2.1232
    alpha = (3.7998, 0.46089, 1.2520, 7.8147, 1.0313, 5.3757, 6.3305, 1.2948)
    eta0 = (1089.7, 1187.7, 797.78, 1058.2, 446.13, 443.0, 465.8, 1026.7)
    sigma = (31.454, 100.46, 91.601, 63.153, 275.111, 45.22, 22.68, 232.14)
    return alpha, eta0, sigma, eps_inf


def _gc(eta, alpha, eta0, sigma):
    """Imaginary part of the dielectric function (Gaussian oscillator sum)."""
    out = np.zeros_like(eta)
    for a, e0, s in zip(alpha, eta0, sigma):
        out += a * np.exp(-4.0 * np.log(2.0) * ((eta - e0) / s) ** 2) - a * np.exp(
            -4.0 * np.log(2.0) * ((eta + e0) / s) ** 2
        )
    return out


def _gc_kkg(eta, alpha, eta0, sigma):
    """Real-part contribution via the Kramers-Kronig-consistent Dawson term.

    Note: the original Kitamura publication has a misprint — ``pi`` in the
    denominator where it should be ``sqrt(pi)``; the correct ``sqrt(pi)`` is
    used here.
    """
    out = np.zeros_like(eta)
    fac = 2.0 * np.sqrt(np.log(2.0))
    for a, e0, s in zip(alpha, eta0, sigma):
        out += (
            2.0
            / np.sqrt(np.pi)
            * a
            * (dawsn(fac * (eta + e0) / s) - dawsn(fac * (eta - e0) / s))
        )
    return out


def _eps(eta, alpha, eta0, sigma, eps_inf):
    return (
        eps_inf * np.ones_like(eta)
        + _gc_kkg(eta, alpha, eta0, sigma)
        + 1j * _gc(eta, alpha, eta0, sigma)
    )


def _load_sampled_nk():
    """Load and sort the sampled n / k CSVs covering 0-15 um.

    Returns ``(n_arr, k_arr)``, each a ``(2, N)`` array of
    ``[wavelength_um; value]`` sorted by wavelength.
    """

    def _stack(file_list):
        lam, val = [], []
        for fname in file_list:
            arr = np.genfromtxt(
                os.path.join(_SILICA_DATA_DIR, fname), delimiter=",", dtype=float
            )
            lam += [row[0] for row in arr]
            val += [row[1] for row in arr]
        lam = np.array(lam)
        val = np.array(val)
        ind = np.argsort(lam)
        return np.array([lam[ind], val[ind]])

    n_arr = _stack(["n_0-1um.csv", "n_1-15um.csv"])
    k_arr = _stack(["k_0-1um.csv", "k_1-15um.csv"])
    return n_arr, k_arr


def _refractive_index(cfg: OpticalConfig):
    """Stage 1: build silica ``(vlambda_vect, n_vect, k_vect)``.

    Returns the log-spaced wavelength grid and the corrected refractive
    index: the Kitamura analytic model, replaced below 8 um by interpolated
    sampled n / k.
    """
    vlambda_vect = np.exp(
        np.linspace(
            np.log(cfg.ri_lambda_min), np.log(cfg.ri_lambda_max), cfg.ri_n_lambda
        )
    )
    eta = 1e4 / vlambda_vect  # wavenumber (cm^-1)

    n_arr, k_arr = _load_sampled_nk()
    alpha, eta0, sigma, eps_inf = _kitamura_params()
    eps_vect = _eps(eta, alpha, eta0, sigma, eps_inf)

    n_vect = np.sqrt(0.5 * (np.real(eps_vect) + np.abs(eps_vect)))
    k_vect = np.imag(eps_vect) / (2.0 * n_vect)

    # Below 8 um: replace the analytic model with sampled measurements.
    ind = np.where(eta >= 1e4 / _LAMBDA_ANALYTIC_MIN_UM)[0]
    n_vect[ind] = np.interp(1e4 / eta[ind], n_arr[0, :], n_arr[1, :])
    k_vect[ind] = np.interp(1e4 / eta[ind], k_arr[0, :], k_arr[1, :])

    return vlambda_vect, n_vect, k_vect


def generate(
    optical_csv: str = DEFAULT_OPTICAL_CSV,
    out_file: str = DEFAULT_OUT_FILE,
    cfg: OpticalConfig | None = None,
) -> str:
    """Build the silica climlab optical table and stamp it.

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
