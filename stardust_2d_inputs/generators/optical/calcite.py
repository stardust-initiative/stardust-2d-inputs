"""optical.calcite — calcite optical-table generator.

Two-stage pipeline for calcite (CaCO3):

  Stage 1 — build the complex refractive index ``m(lambda) = n + i*k`` from a
            Lorentz multi-oscillator dielectric model. Calcite is uniaxially
            birefringent: the published table is the "ray average" — the mean
            of the ordinary-ray and extraordinary-ray dielectric functions —
            which is the isotropic-equivalent index the Mie stage expects.
            This stage is purely analytic (no raw data files).
  Stage 2 — Mie + RRTMG-band averaging into the climlab optical table
            (see :mod:`optical._common`).

Reference
    Long, Querry, Bell & Alexander, "Optical properties of calcite and gypsum
    in crystalline and powdered form in the infrared and far-infrared",
    Infrared Physics 34(2), pp. 191-201, 1993 (Tables 1 and 2 — the ordinary
    and extraordinary ray oscillator parameters).

Configuration is via :class:`optical._common.OpticalConfig`; the output is
``stardust_particles_calcite_AvgRay_climlab.nc``, stamped via
:func:`core.provenance.stamp`.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import OpticalConfig, build_optical_table

GENERATOR = "optical.calcite"

MATERIAL_NAME = "calcite"
DEFAULT_OPTICAL_CSV = "./calcite_optical_RayAvg.csv"
DEFAULT_OUT_FILE = "./stardust_particles_calcite_AvgRay_climlab.nc"

SOURCE = (
    "Refractive-index input: Long, Querry, Bell & Alexander, 'Optical "
    "properties of calcite and gypsum in crystalline and powdered form in "
    "the infrared and far-infrared', Infrared Physics 34(2), 191-201 (1993), "
    "doi:10.1016/0020-0891(93)90008-U. Used as the calcite ordinary-ray and "
    "extraordinary-ray Lorentz-oscillator parameters (Tables 1 and 2); no "
    "raw data files (analytic oscillator model)."
)
PROCESSING = (
    "Stage 1: Lorentz multi-oscillator dielectric model, averaged over the "
    "ordinary and extraordinary rays. Stage 2: Mie efficiencies (miepython) "
    "Planck-weighted into the RRTMG shortwave/longwave band tables."
)

# Lorentz-oscillator parameters from Long et al. (1993). eps_inf is the
# high-frequency permittivity; A / gamma / omega0 are the oscillator
# strengths, damping widths, and resonance wavenumbers (cm^-1).
_RAY_PARAMS = {
    # Table 1 — ordinary ray.
    "OrdinaryRay": dict(
        eps_inf=2.625,
        A=(2.7751, 0.9400, 1.6657, 0.0175, 0.5507),
        gamma=(6.037, 12.051, 13.507, 7.517, 8.519),
        omega0=(104.1, 223.3, 295.6, 713.2, 1406.8),
    ),
    # Table 2 — extraordinary ray.
    "ExtraOrdinaryRay": dict(
        eps_inf=2.170,
        A=(4.6412, 1.3350, 0.0817),
        gamma=(6.797, 10.451, 1.560),
        omega0=(94.6, 304.9, 870.8),
    ),
}


def _eps(omega, params):
    """Lorentz multi-oscillator dielectric function for one ray."""
    eps = params["eps_inf"] * np.ones(omega.shape, dtype=complex)
    for A, gamma, omega0 in zip(params["A"], params["gamma"], params["omega0"]):
        eps += A * omega0**2 / ((omega0**2 - omega**2) - 1j * omega * gamma)
    return eps


def _refractive_index(cfg: OpticalConfig):
    """Stage 1: build ray-averaged calcite ``(vlambda_vect, n_vect, k_vect)``.

    The dielectric function is averaged over the ordinary and extraordinary
    rays before the index is extracted.
    """
    vlambda_vect = np.exp(
        np.linspace(
            np.log(cfg.ri_lambda_min), np.log(cfg.ri_lambda_max), cfg.ri_n_lambda
        )
    )
    omega = 1e4 / vlambda_vect  # wavenumber (cm^-1)

    eps_vect = np.zeros(omega.shape, dtype=complex)
    for params in _RAY_PARAMS.values():
        eps_vect += _eps(omega, params)
    eps_vect /= len(_RAY_PARAMS)

    n_vect = np.sqrt(0.5 * (np.real(eps_vect) + np.abs(eps_vect)))
    k_vect = np.imag(eps_vect) / (2.0 * n_vect)
    return vlambda_vect, n_vect, k_vect


def generate(
    optical_csv: str = DEFAULT_OPTICAL_CSV,
    out_file: str = DEFAULT_OUT_FILE,
    cfg: OpticalConfig | None = None,
) -> str:
    """Build the calcite climlab optical table and stamp it.

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
