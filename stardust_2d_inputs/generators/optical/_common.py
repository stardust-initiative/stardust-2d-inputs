"""optical._common — shared Mie / RRTMG-band machinery for the optical generators.

The optical-table pipeline runs in two stages:

  Stage 1 — per-material refractive-index generation. Each material module
            (:mod:`optical.silica`, :mod:`optical.sulfate`, :mod:`optical.calcite`)
            builds a complex refractive-index table ``m(lambda) = n + i*k`` on a
            wavelength grid from its published spectroscopic sources, and writes
            it to an intermediate ``<material>_optical.csv``.

  Stage 2 — the Mie + RRTMG-band machinery in this module. The intermediate
            refractive-index CSV is loaded into a :class:`Particle`, Mie
            efficiencies are computed per radius with ``miepython``, and the
            spectral quantities are Planck-weighted into the RRTMG shortwave and
            longwave band tables. :class:`RadiationTransferTableRRTMG` drives the
            radius loop and writes the stamped ``stardust_particles_*_climlab.nc``
            output.

Helpers in this module:

  * :class:`Particle` — holds a material's refractive-index spectrum and runs
    the Mie calculation for a given particle radius.
  * :func:`planck_wavelength` — the Planck blackbody flux density in wavelength
    space, used as the spectral weighting function for band averaging.
  * :func:`band_temperature_fit` — fits a per-band effective blackbody
    temperature so the Planck-weighted band fluxes match a target weight
    distribution (the longwave RRTMG band weights).
  * :class:`RadiationTransferTableRRTMG` — the Stage-2 table builder.

The intermediate refractive-index CSV is read from / written to a path supplied
by the caller (config-driven); nothing in this module hard-codes filesystem
locations. This module needs the ``[generators]`` extra (``miepython``, ``scipy``,
``pandas``); it is not imported at simulation runtime.
"""

from __future__ import annotations

import os
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import netCDF4 as ncdf4
import pandas as pd
import miepython
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

from ...core import provenance

# miepython renamed its efficiency entry point across major versions
# (``mie`` -> ``efficiencies_mx``). Resolve whichever this install provides;
# both return ``(qext, qsca, qback, g)`` for ``(m, x)`` array inputs.
if hasattr(miepython, "efficiencies_mx"):
    _mie_efficiencies = miepython.efficiencies_mx
elif hasattr(miepython, "mie"):
    _mie_efficiencies = miepython.mie
else:  # pragma: no cover - defensive
    raise ImportError(
        "miepython provides neither 'efficiencies_mx' nor 'mie'; "
        "an unsupported version is installed."
    )

# Repository root — the git state stamped into generator outputs is that of
# this engine repo (provenance.git_describe walks up from here).
_REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Directory holding the carried raw refractive-index source files, one
# subdirectory per material (optical/data/<material>/).
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Physical constants ──
K_BOLTZMANN = 1.3806488e-23  # Boltzmann constant (J / K)
C_LIGHT = 2.99792458e8  # speed of light (m / s)
H_PLANCK = 6.62606957e-34  # Planck constant (J s)


# ═══════════════════════════════════════════════════
#  Stage-2 input: a material's refractive-index spectrum
# ═══════════════════════════════════════════════════


@dataclass
class Particle:
    """A material's refractive-index spectrum plus its Mie efficiencies.

    A ``Particle`` is constructed from an intermediate refractive-index CSV
    (the Stage-1 output: columns ``Wavelength (um)``, ``Real Part``,
    ``Imaginary Part``). The complex index ``m(lambda)`` is interpolated onto
    the requested wavelength grid. :meth:`mie_calc` then computes the Mie
    extinction / scattering / absorption efficiencies for a given particle
    radius via ``miepython``.
    """

    rho: float  # bulk density (g / cm^3)
    lambda_um_vect: np.ndarray  # wavelength grid (micron)
    r_nanoparticle_um: float  # particle radius (micron)
    m_vect: np.ndarray  # complex refractive index n + i*k
    material_name: str
    optical_csv_path: str

    q_ext: np.ndarray = field(default=None)
    q_sca: np.ndarray = field(default=None)
    q_abs: np.ndarray = field(default=None)
    g: np.ndarray = field(default=None)
    beta_bar: np.ndarray = field(default=None)
    beta_top: np.ndarray = field(default=None)
    beta_mu: np.ndarray = field(default=None)
    mu_vect: np.ndarray = field(default=None)

    # Bulk density per material (g / cm^3). Carried over unchanged from the
    # source pipeline; only the three migrated materials are listed.
    _DENSITY = {
        "silica": 2.196,
        "sulfate": 1.69,
        "calcite": 2.711,
    }

    def __init__(
        self,
        material_name,
        lambda_um_vect,
        optical_csv_path,
        r_nanoparticle_um=0.0,
        ntt=1000,
        mu_vect=None,
    ):
        self.material_name = material_name
        self.lambda_um_vect = lambda_um_vect
        self.r_nanoparticle_um = r_nanoparticle_um
        self.optical_csv_path = optical_csv_path

        key = material_name.lower()
        if key not in self._DENSITY:
            raise ValueError(
                f"{material_name}: unknown material "
                f"(supported: {sorted(self._DENSITY)})"
            )
        self.rho = self._DENSITY[key]

        df = pd.read_csv(optical_csv_path)
        self.m_vect = np.zeros(lambda_um_vect.shape, dtype=complex)
        self.m_vect.real = np.interp(
            lambda_um_vect, df["Wavelength (um)"], df["Real Part"]
        )
        self.m_vect.imag = np.interp(
            lambda_um_vect, df["Wavelength (um)"], df["Imaginary Part"]
        )

        if self.r_nanoparticle_um > 0.0:
            self.mie_calc(self.r_nanoparticle_um, ntt=ntt, mu_vect=mu_vect)

    def mie_calc(self, r_nanoparticle_um, ntt=1000, mu_vect=None):
        """Compute Mie efficiencies for a particle of radius ``r_nanoparticle_um``.

        Fills ``q_ext``, ``q_sca``, ``q_abs``, ``g`` (asymmetry factor) and,
        when ``ntt > 0``, the angular-integral quantities ``beta_top`` /
        ``beta_bar`` / ``beta_mu`` used in the shortwave band averaging.
        """
        x = 2.0 * np.pi * r_nanoparticle_um / self.lambda_um_vect
        n_lam = len(x)

        if mu_vect is not None:
            n_theta = len(mu_vect)
            beta_mu = np.zeros((n_lam, n_theta))
        else:
            n_theta = 0
            beta_mu = None

        if ntt > 0:
            beta_top = np.zeros_like(x)
            beta_bar = np.zeros_like(x)
            for i_lam in range(n_lam):
                m = self.m_vect[i_lam]
                bt, bb, bmu = self.beta_calc(m, x[i_lam], ntt=ntt, mu_vect=mu_vect)
                beta_top[i_lam] = bt
                beta_bar[i_lam] = bb
                if n_theta > 0:
                    beta_mu[i_lam, :] = bmu
        else:
            beta_top = None
            beta_bar = None

        # miepython's convention assumes imag(m) < 0
        q_ext, q_sca, _, g = _mie_efficiencies(np.conjugate(self.m_vect), x)

        self.q_ext = q_ext
        self.q_sca = q_sca
        self.q_abs = q_ext - q_sca
        self.g = g
        self.beta_bar = beta_bar
        self.beta_top = beta_top
        self.mu_vect = mu_vect
        self.beta_mu = beta_mu

    @staticmethod
    def beta_calc(m, x, ntt=1000, mu_vect=None):
        """Angular integrals of the Mie phase function for one ``(m, x)``.

        Returns ``(beta_top, beta_bar, beta_mu)``: the backward-hemisphere
        fraction, the mean-cosine-weighted fraction, and (when ``mu_vect`` is
        given) the per-zenith-angle fraction used in the shortwave averaging.
        """
        mu_tag_vect = np.linspace(-1.0, 1.0, 2 * ntt)
        dt = 2.0 / (2 * ntt - 1)

        if mu_vect is not None:
            n_theta = len(mu_vect)
            beta_mu = np.zeros((n_theta,))
        else:
            n_theta = 0
            beta_mu = None

        beta_top = beta_bar = None
        if ntt > 0:
            ind = np.where(mu_tag_vect <= 0)[0]
            m_miepython = np.conjugate(m)  # miepython convention: imag(m) < 0
            P = miepython.i_unpolarized(m_miepython, x, mu_tag_vect, "4pi")
            beta_top = 0.5 * np.trapz(P[ind], mu_tag_vect[ind])
            beta_bar = (
                1.0 / (2.0 * np.pi) * np.trapz(P * np.arccos(mu_tag_vect), mu_tag_vect)
            )

        if n_theta > 0:
            P_mu_func = interp1d(mu_tag_vect, P, kind="quadratic")
            for i_mu, mu in enumerate(mu_vect):
                dbeta_mu1, dbeta_mu2, dbeta_mu3 = 0.0, 0.0, 0.0
                theta = np.arccos(mu)
                i1_1_float, i1_2 = (1.0 + np.sin(theta)) / dt, 2 * ntt - 1
                i3_1, i3_2_float = 0, (1.0 - np.sin(theta)) / dt

                arr_sin_t = np.array([np.sin(theta)])
                arr_sin_mt = np.array([-np.sin(theta)])
                if mu == 0.0:
                    beta_mu[i_mu] = 0.5
                else:
                    mu_tag_1 = mu_tag_vect[
                        np.arange(int(np.ceil(i1_1_float)), i1_2 + 1)
                    ]
                    if len(mu_tag_1) > 0:
                        if mu_tag_1[0] > np.sin(theta):
                            mu_tag_1 = np.concatenate((arr_sin_t, mu_tag_1))
                        dbeta_mu1 = np.trapz(P_mu_func(mu_tag_1), mu_tag_1)

                    mu_tag_3 = mu_tag_vect[np.arange(i3_1, int(np.floor(i3_2_float)))]
                    if len(mu_tag_3) > 0:
                        if mu_tag_3[-1] < -np.sin(theta):
                            mu_tag_3 = np.concatenate((mu_tag_3, arr_sin_mt))
                        dbeta_mu3 = np.trapz(-P_mu_func(mu_tag_3), mu_tag_3)

                    if theta > 0.0:
                        mu_tag_2 = mu_tag_vect[
                            np.arange(
                                int(np.ceil(i3_2_float)), int(np.floor(i1_1_float))
                            )
                        ]
                        if len(mu_tag_2) > 0:
                            if mu_tag_2[0] > -np.sin(theta):
                                mu_tag_2 = np.concatenate((arr_sin_mt, mu_tag_2))
                            if mu_tag_2[-1] < np.sin(theta):
                                mu_tag_2 = np.concatenate((mu_tag_2, arr_sin_t))
                            theta_tag_vect2 = np.arccos(mu_tag_2)
                            denom = 1.0 / (np.tan(theta) * np.tan(theta_tag_vect2))
                            denom = np.clip(denom, -1.0, 1.0)
                            dbeta_mu2 = np.trapz(
                                (1.0 - 2.0 / np.pi * np.arccos(denom))
                                * P_mu_func(mu_tag_2),
                                mu_tag_2,
                            )
                    beta_mu[i_mu] = 0.5 - 0.25 * (dbeta_mu1 + dbeta_mu2 + dbeta_mu3)
        return beta_top, beta_bar, beta_mu


# ═══════════════════════════════════════════════════
#  Planck function and longwave band-temperature fit
# ═══════════════════════════════════════════════════


def planck_wavelength(wavelength_m, temperature_k):
    """Planck blackbody flux density in wavelength space.

    ``wavelength_m`` is wavelength in metres, ``temperature_k`` in kelvin.
    Formula (3.3) of Pierrehumbert, *Principles of Planetary Climate*.
    """
    u = H_PLANCK * C_LIGHT / wavelength_m / K_BOLTZMANN / temperature_k
    return (
        2.0
        * K_BOLTZMANN**5
        * temperature_k**5
        / H_PLANCK**4
        / C_LIGHT**3
        * u**5
        / (np.exp(u) - 1.0)
    )


def _integrated_weight_func(xmin=1e-2, xmax=20.0, n=1000):
    """Build the cumulative blackbody-weight function ``f(x)`` and its derivative.

    ``x = h*c/(lambda*k*T)`` is the dimensionless Planck argument; ``f(x)`` is
    the fraction of blackbody flux below ``x``. Returns ``(f, dfdx)`` as
    callables, with small-/large-``x`` analytic continuations outside the grid.
    """
    x_vect = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    f_small = lambda x: 15.0 / np.pi**4 * (x**3 / 3.0 - x**4 / 8.0)
    f_large = lambda x: 1.0 - 15.0 / np.pi**4 * np.exp(-x) * (
        x**3 + 3.0 * x**2 + 6.0 * x + 6.0
    )
    f_vect = f_small(xmin) + cumulative_trapezoid(
        15.0 / np.pi**4 * x_vect**3 * np.exp(-x_vect) / (1.0 - np.exp(-x_vect)),
        x_vect,
        initial=0.0,
    )
    f_mid = lambda x: np.interp(x, x_vect, f_vect)

    dfdx_small = lambda x: 15.0 / np.pi**4 * x**2 * (1.0 - 0.5 * x)
    dfdx_large = lambda x: 15.0 / np.pi**4 * x**3 * np.exp(-x)
    dfdx_mid = lambda x: (15.0 / np.pi**4 * x**3 * np.exp(-x) / (1.0 - np.exp(-x)))

    f = lambda x: np.where(
        x > xmin, np.where(x < xmax, f_mid(x), f_large(x)), f_small(x)
    )
    dfdx = lambda x: np.where(
        x > xmin, np.where(x < xmax, dfdx_mid(x), dfdx_large(x)), dfdx_small(x)
    )
    return f, dfdx


def band_temperature_fit(
    wavenumber_group_bounds,
    weight_target,
    temp_guess=None,
    xmin=1e-2,
    xmax=20.0,
    n=1000,
    temp_min=150.0,
    temp_max=300.0,
):
    """Fit a per-band effective blackbody temperature for the longwave bands.

    Each longwave RRTMG band carries a fraction of the total blackbody flux.
    This routine solves for the per-band temperature vector whose
    Planck-weighted band fluxes best match ``weight_target`` (a least-squares
    fit, L-BFGS-B). Returns ``(temperature_vector, achieved_weights)``.
    """
    f, dfdx = _integrated_weight_func(xmin=xmin, xmax=xmax, n=n)
    ng = len(wavenumber_group_bounds) - 1
    if len(weight_target) != ng:
        raise ValueError("len(weight_target) must equal the number of groups")
    if temp_guess is None or np.size(temp_guess) == 0:
        temp_guess = 260.0 * np.ones((ng,))

    # gamma * wavenumber / T is dimensionless
    gamma = H_PLANCK * 100.0 * C_LIGHT / K_BOLTZMANN

    def _band_fluxes(t):
        a = np.zeros((ng,))
        for k, t0 in enumerate(t):
            wk = wavenumber_group_bounds[k]
            wkp1 = wavenumber_group_bounds[k + 1]
            a[k] = t0**4 * (f(gamma * wkp1 / t0) - f(gamma * wk / t0))
        return a

    def err_func(t):
        a = _band_fluxes(t)
        weight = a / np.sum(a)
        return 0.5 * np.sum((weight - weight_target) ** 2)

    def derr_func(t):
        a = np.zeros((ng,))
        da = np.zeros((ng,))
        for k, t0 in enumerate(t):
            wk = wavenumber_group_bounds[k]
            wkp1 = wavenumber_group_bounds[k + 1]
            a[k] = t0**4 * (f(gamma * wkp1 / t0) - f(gamma * wk / t0))
            da[k] = 4.0 * t0**3 * (
                f(gamma * wkp1 / t0) - f(gamma * wk / t0)
            ) - gamma * t0**2 * (
                dfdx(gamma * wkp1 / t0) * wkp1 - dfdx(gamma * wk / t0) * wk
            )
        b = np.sum(a)
        weight = a / b
        dweight = da / b - a * da / b**2
        return (weight - weight_target) * dweight

    bounds = [(temp_min, temp_max)] * ng
    options = {"disp": False, "gtol": 1e-7}
    res = minimize(
        err_func,
        temp_guess,
        method="L-BFGS-B",
        jac=derr_func,
        bounds=bounds,
        options=options,
    )
    temp = res.x
    a = _band_fluxes(temp)
    weight = a / np.sum(a)
    return temp, weight


# ═══════════════════════════════════════════════════
#  Stage-2 builder: Mie + RRTMG-band optical tables
# ═══════════════════════════════════════════════════

# RRTMG longwave band boundaries in cm^-1 (16 bands).
RRTMG_LW_BAND_BOUNDS = np.array(
    [
        10,
        350,
        500,
        630,
        700,
        820,
        980,
        1080,
        1180,
        1390,
        1480,
        1800,
        2080,
        2250,
        2390,
        2600,
        3250,
    ]
)

# Target longwave band weights (fractional blackbody flux per band) used by
# band_temperature_fit to derive the per-band effective temperatures.
RRTMG_LW_WEIGHT_TARGET = np.array(
    [
        1.30318388e-01,
        1.53140286e-01,
        1.38806296e-01,
        3.75508366e-02,
        1.23016844e-01,
        1.69188702e-01,
        8.69503585e-02,
        6.73191260e-02,
        6.08813159e-02,
        7.20118029e-03,
        8.73154406e-03,
        8.78221684e-03,
        3.44513991e-03,
        4.75064869e-05,
        2.70627632e-03,
        1.91398337e-03,
    ]
)


# ── Parallel radius-loop machinery ──
#
# The per-radius work is fully independent across radii. To distribute it with
# a ``ProcessPoolExecutor`` we (a) build the radius-independent context once per
# worker process in an initializer, storing it in a module-level global, and
# (b) map a self-contained per-radius worker over ``enumerate(r_m_vect)``,
# reassembling the returned rows back into the result arrays by radius index.
#
# The math inside a single radius is byte-for-byte identical to the serial
# code: same ``mie_calc`` calls, same ``np.trapz`` band integrations in the
# same order. The ONLY change is which process runs a given radius.

# Per-worker context, populated by ``_radius_worker_init``.
_WORKER_CTX = None


def _build_radius_context(
    particle_sw,
    particle_lw,
    mu_vect,
    n_theta_tag,
    nu_wn_sw1,
    nu_wn_sw2,
    nu_wn_lw,
    T_sw_vect,
    T_lw_vect,
    wavenumber_sw_bound_vect,
    wavenumber_lw_bound_vect,
    lambda_um_sw_vect,
    lambda_um_lw_vect,
    n_mu,
):
    """Assemble the radius-independent context shared by every radius.

    Computes, once, the per-pass band bounds, effective temperatures, the
    Planck spectral weighting ``p_spec_vect`` and the wavelength/wavenumber
    grids for both the LW (``do_sw=False``) and SW (``do_sw=True``) passes.
    These are exactly the quantities the serial ``_build`` computed outside the
    radius loop. Returns a dict consumed by :func:`_compute_radius`.
    """
    passes = {}
    for do_sw in (False, True):
        if do_sw:
            nu1_wn, nu2_wn = nu_wn_sw1, nu_wn_sw2
            T_vect = T_sw_vect
            wavenumber_bound_vect = wavenumber_sw_bound_vect
            lambda_um_vect = lambda_um_sw_vect
        else:
            nu1_wn, nu2_wn = nu_wn_lw[:-1], nu_wn_lw[1:]
            T_vect = T_lw_vect
            wavenumber_bound_vect = wavenumber_lw_bound_vect
            lambda_um_vect = lambda_um_lw_vect

        wavenumber_vect = 1e4 / lambda_um_vect

        # Planck spectral weighting (per-band effective temperature).
        p_spec_vect = np.zeros_like(wavenumber_vect)
        for i in range(len(T_vect)):
            ind = np.where(
                (wavenumber_vect >= wavenumber_bound_vect[i])
                & (wavenumber_vect < wavenumber_bound_vect[i + 1])
            )[0]
            wavelength_m = 1e-2 / wavenumber_vect[ind]
            p_spec_vect[ind] = planck_wavelength(wavelength_m, T_vect[i])
        p_spec_vect /= np.trapz(p_spec_vect, lambda_um_vect)

        passes[do_sw] = {
            "nu1_wn": nu1_wn,
            "nu2_wn": nu2_wn,
            "lambda_um_vect": lambda_um_vect,
            "wavenumber_vect": wavenumber_vect,
            "p_spec_vect": p_spec_vect,
        }

    return {
        "particle_sw": particle_sw,
        "particle_lw": particle_lw,
        "mu_vect": mu_vect,
        "n_theta_tag": n_theta_tag,
        "n_mu": n_mu,
        "passes": passes,
    }


def _compute_radius(
    ir, r_m, ctx, *, verbose=False, material_name="", nr=0, rmin_nm=0.0, rmax_nm=0.0
):
    """Compute every band row for a single radius (both LW and SW passes).

    Returns ``(ir, rows)`` where ``rows`` is a dict of the per-radius row
    arrays keyed by output-array name. The float operations are performed in
    exactly the same order as the serial code, so the result is byte-for-byte
    identical regardless of which process runs it.
    """
    n_sw = len(ctx["passes"][True]["nu1_wn"])
    n_lw = len(ctx["passes"][False]["nu1_wn"])
    n_mu = ctx["n_mu"]

    rows = {
        "babs_lw": np.zeros((n_lw,)),
        "bext_sw": np.zeros((n_sw,)),
        "bsca_sw": np.zeros((n_sw,)),
        "basc_sw": np.zeros((n_sw,)),
        "bet_bar_bsca_sw": np.zeros((n_sw,)),
        "bet_mu_bsca_sw": np.zeros((n_sw, n_mu)),
    }

    for do_sw in (False, True):
        p = ctx["passes"][do_sw]
        nu1_wn, nu2_wn = p["nu1_wn"], p["nu2_wn"]
        lambda_um_vect = p["lambda_um_vect"]
        wavenumber_vect = p["wavenumber_vect"]
        p_spec_vect = p["p_spec_vect"]

        if verbose:
            print(
                f"{material_name}: radius {ir + 1}/{nr} "
                f"r={r_m * 1e9:.1f} nm "
                f"(rmin={rmin_nm}, rmax={rmax_nm})"
            )

        r_um = r_m * 1e6
        fac = np.pi * r_m**2  # geometric cross-section (m^2)

        if do_sw:
            particle_sw = ctx["particle_sw"]
            particle_sw.mie_calc(
                r_nanoparticle_um=r_um, ntt=ctx["n_theta_tag"], mu_vect=ctx["mu_vect"]
            )
            Qabs_vlambda = particle_sw.q_abs * fac
            Qsca_vlambda = particle_sw.q_sca * fac
            Qasc_vlambda = particle_sw.g * particle_sw.q_sca * fac
        else:
            particle_lw = ctx["particle_lw"]
            particle_lw.mie_calc(r_nanoparticle_um=r_um, ntt=0, mu_vect=None)
            Qabs_vlambda = particle_lw.q_abs * fac

        for ig, (w1, w2) in enumerate(zip(nu1_wn, nu2_wn)):
            ind_g = np.where((wavenumber_vect >= w1) & (wavenumber_vect <= w2))[0]
            ref_spec_weight = np.trapz(p_spec_vect[ind_g], lambda_um_vect[ind_g])
            qabs_Lambda = (
                np.trapz(
                    Qabs_vlambda[ind_g] * p_spec_vect[ind_g], lambda_um_vect[ind_g]
                )
                / ref_spec_weight
            )

            if do_sw:
                bbar_qsca_Lambda = (
                    np.trapz(
                        (particle_sw.beta_bar * Qsca_vlambda)[ind_g]
                        * p_spec_vect[ind_g],
                        lambda_um_vect[ind_g],
                    )
                    / ref_spec_weight
                )
                rows["bet_bar_bsca_sw"][ig] = bbar_qsca_Lambda
                qsca_Lambda = (
                    np.trapz(
                        Qsca_vlambda[ind_g] * p_spec_vect[ind_g], lambda_um_vect[ind_g]
                    )
                    / ref_spec_weight
                )
                qasc_Lambda = (
                    np.trapz(
                        Qasc_vlambda[ind_g] * p_spec_vect[ind_g], lambda_um_vect[ind_g]
                    )
                    / ref_spec_weight
                )
                rows["bext_sw"][ig] = qsca_Lambda + qabs_Lambda
                rows["bsca_sw"][ig] = qsca_Lambda
                rows["basc_sw"][ig] = qasc_Lambda
            else:
                rows["babs_lw"][ig] = qabs_Lambda

        if do_sw:
            for i_mu in range(n_mu):
                for ig, (w1, w2) in enumerate(zip(nu1_wn, nu2_wn)):
                    ind_g = (wavenumber_vect >= w1) & (wavenumber_vect <= w2)
                    ref_spec_weight = np.trapz(
                        p_spec_vect[ind_g], lambda_um_vect[ind_g]
                    )
                    rows["bet_mu_bsca_sw"][ig, i_mu] = (
                        np.trapz(
                            (particle_sw.beta_mu[:, i_mu] * Qsca_vlambda)[ind_g]
                            * p_spec_vect[ind_g],
                            lambda_um_vect[ind_g],
                        )
                        / ref_spec_weight
                    )

    return ir, rows


def _radius_worker_init(
    particle_sw,
    particle_lw,
    mu_vect,
    n_theta_tag,
    nu_wn_sw1,
    nu_wn_sw2,
    nu_wn_lw,
    T_sw_vect,
    T_lw_vect,
    wavenumber_sw_bound_vect,
    wavenumber_lw_bound_vect,
    lambda_um_sw_vect,
    lambda_um_lw_vect,
    n_mu,
):
    """ProcessPool initializer: build the per-worker radius context once.

    Stores the assembled context (including the worker's own ``Particle``
    objects) in the module-level ``_WORKER_CTX`` so each radius task can reuse
    it without re-pickling per call.
    """
    global _WORKER_CTX
    _WORKER_CTX = _build_radius_context(
        particle_sw,
        particle_lw,
        mu_vect,
        n_theta_tag,
        nu_wn_sw1,
        nu_wn_sw2,
        nu_wn_lw,
        T_sw_vect,
        T_lw_vect,
        wavenumber_sw_bound_vect,
        wavenumber_lw_bound_vect,
        lambda_um_sw_vect,
        lambda_um_lw_vect,
        n_mu,
    )


def _radius_worker_task(args):
    """ProcessPool task wrapper: compute one radius using ``_WORKER_CTX``."""
    ir, r_m = args
    return _compute_radius(ir, r_m, _WORKER_CTX)


class RadiationTransferTableRRTMG:
    """Stage-2 RRTMG optical-table builder for a single material.

    Loads a material's intermediate refractive-index CSV into shortwave and
    longwave :class:`Particle` objects, runs the Mie calculation over a
    log-spaced radius grid, and Planck-weights the spectral efficiencies into
    the RRTMG shortwave (14-band) and longwave (16-band) tables. The result is
    written to a NetCDF table via :meth:`generate_nc_file`.
    """

    # RRTMG shortwave band boundaries in cm^-1 (14 bands; lower / upper).
    NU_WN_SW1 = np.array(
        [
            2600.0,
            3250.0,
            4000.0,
            4650.0,
            5150.0,
            6150.0,
            7700.0,
            8050.0,
            12850.0,
            16000.0,
            22650.0,
            29000.0,
            38000.0,
            820.0,
        ]
    )
    NU_WN_SW2 = np.array(
        [
            3250.0,
            4000.0,
            4650.0,
            5150.0,
            6150.0,
            7700.0,
            8050.0,
            12850.0,
            16000.0,
            22650.0,
            29000.0,
            38000.0,
            50000.0,
            2600.0,
        ]
    )

    def __init__(
        self,
        material_name,
        lambda_um_sw_vect,
        lambda_um_lw_vect,
        optical_csv_path,
        *,
        T_sw=6000.0,
        wavenumber_lw_bound_list=(600.0, 800.0, 1250.0),
        T_lw_list=(255.0, 220.0, 290.0, 255.0),
        rmin_nm=0.4,
        rmax_nm=1500.0,
        nr=100,
        n_theta_tag=1000,
        n_mu=100,
        mu_min=0.0,
        mu_max=1.0,
        n_workers=1,
    ):
        self.material_name = material_name
        self.T_sw_vect = np.array([T_sw])
        self.T_lw_vect = np.array(T_lw_list)
        self.wavenumber_lw_bound_vect = np.concatenate(
            ([-np.inf], np.array(wavenumber_lw_bound_list), [np.inf])
        )
        self.wavenumber_sw_bound_vect = np.array([-np.inf, np.inf])
        if len(self.T_lw_vect) != len(self.wavenumber_lw_bound_vect) - 1:
            raise ValueError(
                "len(T_lw_list) must equal len(wavenumber_lw_bound_list) + 1"
            )

        self.nu_wn_lw = RRTMG_LW_BAND_BOUNDS.astype(float)
        self.nu_wn_sw1 = self.NU_WN_SW1
        self.nu_wn_sw2 = self.NU_WN_SW2

        self.particle_lw = Particle(material_name, lambda_um_lw_vect, optical_csv_path)
        self.particle_sw = Particle(material_name, lambda_um_sw_vect, optical_csv_path)
        self.logr_vect = np.linspace(np.log(rmin_nm * 1e-9), np.log(rmax_nm * 1e-9), nr)
        self.r_m_vect = np.exp(self.logr_vect)
        self.mu_vect = np.linspace(mu_min, mu_max, n_mu)
        self.n_theta_tag = n_theta_tag

        self.rho_mks = 1e3 * self.particle_sw.rho  # g/cm^3 -> kg/m^3

        n_sw = len(self.nu_wn_sw1)
        n_lw = len(self.nu_wn_lw) - 1

        # Volume-specific coefficients (m^-1 for the climlab table).
        self.babs_lw = np.zeros((nr, n_lw))
        self.bsca_sw = np.zeros((nr, n_sw))
        self.basc_sw = np.zeros((nr, n_sw))
        self.bext_sw = np.zeros((nr, n_sw))
        self.ref_spec_weight_sw = np.zeros((n_sw,))
        self.ref_spec_weight_lw = np.zeros((n_lw,))
        self.bet_bar_bsca_sw = np.zeros((nr, n_sw))
        self.bet_mu_bsca_sw = np.zeros((nr, n_sw, n_mu))

        self._build(
            material_name,
            lambda_um_sw_vect,
            lambda_um_lw_vect,
            nr,
            rmin_nm,
            rmax_nm,
            n_mu,
            n_workers,
        )

    def _build(
        self,
        material_name,
        lambda_um_sw_vect,
        lambda_um_lw_vect,
        nr,
        rmin_nm,
        rmax_nm,
        n_mu,
        n_workers=1,
    ):
        """Run the radius loop, Planck-weighting Mie efficiencies into bands.

        When ``n_workers > 1`` the per-radius work is distributed across a
        ``ProcessPoolExecutor`` (radii are independent and reassembled by
        index). When ``n_workers <= 1`` the original serial code path runs
        unchanged.
        """
        # ── Radius-independent reference spectrum weights (computed once;
        #    identical across radii). ──
        ctx = _build_radius_context(
            self.particle_sw,
            self.particle_lw,
            self.mu_vect,
            self.n_theta_tag,
            self.nu_wn_sw1,
            self.nu_wn_sw2,
            self.nu_wn_lw,
            self.T_sw_vect,
            self.T_lw_vect,
            self.wavenumber_sw_bound_vect,
            self.wavenumber_lw_bound_vect,
            lambda_um_sw_vect,
            lambda_um_lw_vect,
            n_mu,
        )
        for do_sw in (False, True):
            p = ctx["passes"][do_sw]
            for ig, (w1, w2) in enumerate(zip(p["nu1_wn"], p["nu2_wn"])):
                ind_g = np.where(
                    (p["wavenumber_vect"] >= w1) & (p["wavenumber_vect"] <= w2)
                )[0]
                ref_spec_weight = np.trapz(
                    p["p_spec_vect"][ind_g], p["lambda_um_vect"][ind_g]
                )
                if do_sw:
                    self.ref_spec_weight_sw[ig] = ref_spec_weight
                else:
                    self.ref_spec_weight_lw[ig] = ref_spec_weight

        if n_workers > 1:
            self._build_parallel(
                material_name, lambda_um_sw_vect, lambda_um_lw_vect, nr, n_mu, n_workers
            )
            return

        for do_sw in (False, True):
            if do_sw:
                nu1_wn, nu2_wn = self.nu_wn_sw1, self.nu_wn_sw2
                T_vect = self.T_sw_vect
                wavenumber_bound_vect = self.wavenumber_sw_bound_vect
                lambda_um_vect = lambda_um_sw_vect
            else:
                nu_wn = self.nu_wn_lw
                nu1_wn, nu2_wn = nu_wn[:-1], nu_wn[1:]
                T_vect = self.T_lw_vect
                wavenumber_bound_vect = self.wavenumber_lw_bound_vect
                lambda_um_vect = lambda_um_lw_vect

            wavenumber_vect = 1e4 / lambda_um_vect

            # Planck spectral weighting (per-band effective temperature).
            p_spec_vect = np.zeros_like(wavenumber_vect)
            for i in range(len(T_vect)):
                ind = np.where(
                    (wavenumber_vect >= wavenumber_bound_vect[i])
                    & (wavenumber_vect < wavenumber_bound_vect[i + 1])
                )[0]
                wavelength_m = 1e-2 / wavenumber_vect[ind]
                p_spec_vect[ind] = planck_wavelength(wavelength_m, T_vect[i])
            p_spec_vect /= np.trapz(p_spec_vect, lambda_um_vect)

            for ir, r_m in enumerate(self.r_m_vect):
                print(
                    f"{material_name}: radius {ir + 1}/{nr} "
                    f"r={r_m * 1e9:.1f} nm "
                    f"(rmin={rmin_nm}, rmax={rmax_nm})"
                )
                r_um = r_m * 1e6
                fac = np.pi * r_m**2  # geometric cross-section (m^2)

                if do_sw:
                    self.particle_sw.mie_calc(
                        r_nanoparticle_um=r_um,
                        ntt=self.n_theta_tag,
                        mu_vect=self.mu_vect,
                    )
                    Qabs_vlambda = self.particle_sw.q_abs * fac
                    Qsca_vlambda = self.particle_sw.q_sca * fac
                    Qasc_vlambda = self.particle_sw.g * self.particle_sw.q_sca * fac
                else:
                    self.particle_lw.mie_calc(
                        r_nanoparticle_um=r_um, ntt=0, mu_vect=None
                    )
                    Qabs_vlambda = self.particle_lw.q_abs * fac

                for ig, (w1, w2) in enumerate(zip(nu1_wn, nu2_wn)):
                    ind_g = np.where((wavenumber_vect >= w1) & (wavenumber_vect <= w2))[
                        0
                    ]
                    ref_spec_weight = np.trapz(
                        p_spec_vect[ind_g], lambda_um_vect[ind_g]
                    )
                    qabs_Lambda = (
                        np.trapz(
                            Qabs_vlambda[ind_g] * p_spec_vect[ind_g],
                            lambda_um_vect[ind_g],
                        )
                        / ref_spec_weight
                    )

                    if do_sw:
                        bbar_qsca_Lambda = (
                            np.trapz(
                                (self.particle_sw.beta_bar * Qsca_vlambda)[ind_g]
                                * p_spec_vect[ind_g],
                                lambda_um_vect[ind_g],
                            )
                            / ref_spec_weight
                        )
                        self.bet_bar_bsca_sw[ir, ig] = bbar_qsca_Lambda
                        qsca_Lambda = (
                            np.trapz(
                                Qsca_vlambda[ind_g] * p_spec_vect[ind_g],
                                lambda_um_vect[ind_g],
                            )
                            / ref_spec_weight
                        )
                        qasc_Lambda = (
                            np.trapz(
                                Qasc_vlambda[ind_g] * p_spec_vect[ind_g],
                                lambda_um_vect[ind_g],
                            )
                            / ref_spec_weight
                        )
                        self.bext_sw[ir, ig] = qsca_Lambda + qabs_Lambda
                        self.bsca_sw[ir, ig] = qsca_Lambda
                        self.basc_sw[ir, ig] = qasc_Lambda
                        self.ref_spec_weight_sw[ig] = ref_spec_weight
                    else:
                        self.babs_lw[ir, ig] = qabs_Lambda
                        self.ref_spec_weight_lw[ig] = ref_spec_weight

                if do_sw:
                    for i_mu in range(n_mu):
                        for ig, (w1, w2) in enumerate(zip(nu1_wn, nu2_wn)):
                            ind_g = (wavenumber_vect >= w1) & (wavenumber_vect <= w2)
                            ref_spec_weight = np.trapz(
                                p_spec_vect[ind_g], lambda_um_vect[ind_g]
                            )
                            self.bet_mu_bsca_sw[ir, ig, i_mu] = (
                                np.trapz(
                                    (self.particle_sw.beta_mu[:, i_mu] * Qsca_vlambda)[
                                        ind_g
                                    ]
                                    * p_spec_vect[ind_g],
                                    lambda_um_vect[ind_g],
                                )
                                / ref_spec_weight
                            )

    def _build_parallel(
        self, material_name, lambda_um_sw_vect, lambda_um_lw_vect, nr, n_mu, n_workers
    ):
        """Distribute the per-radius work across a ``ProcessPoolExecutor``.

        Each worker builds its own radius context (Particle objects + Planck
        weighting) once via the initializer; radii are mapped over and the
        returned rows are assembled back into ``self.*`` by radius index, so
        the result is independent of completion order.
        """
        init_args = (
            self.particle_sw,
            self.particle_lw,
            self.mu_vect,
            self.n_theta_tag,
            self.nu_wn_sw1,
            self.nu_wn_sw2,
            self.nu_wn_lw,
            self.T_sw_vect,
            self.T_lw_vect,
            self.wavenumber_sw_bound_vect,
            self.wavenumber_lw_bound_vect,
            lambda_um_sw_vect,
            lambda_um_lw_vect,
            n_mu,
        )

        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_radius_worker_init, initargs=init_args
        ) as pool:
            for ir, rows in pool.map(
                _radius_worker_task, list(enumerate(self.r_m_vect))
            ):
                self.babs_lw[ir, :] = rows["babs_lw"]
                self.bext_sw[ir, :] = rows["bext_sw"]
                self.bsca_sw[ir, :] = rows["bsca_sw"]
                self.basc_sw[ir, :] = rows["basc_sw"]
                self.bet_bar_bsca_sw[ir, :] = rows["bet_bar_bsca_sw"]
                self.bet_mu_bsca_sw[ir, :, :] = rows["bet_mu_bsca_sw"]

    def generate_nc_file(self, filename, global_attrs=None):
        """Write the climlab-format RRTMG optical table to ``filename``.

        ``global_attrs`` is an optional mapping of provenance global
        attributes (see :func:`core.provenance.stamp`); when given they are
        written into the same single NetCDF write.
        """

        def str2b(s, n=32):
            s1 = [b" "] * n
            for k in range(len(s)):
                s1[k] = s[k].encode()
            return s1

        name_len = 32
        with ncdf4.Dataset(filename, "w") as dst:
            if global_attrs:
                dst.setncatts({k: v for k, v in global_attrs.items()})
            # ── dimensions ──
            dst.createDimension("temperature", 1)
            dst.createDimension("name_len", name_len)
            dst.createDimension("mu_samples", len(self.logr_vect))
            dst.createDimension("lw_band", self.babs_lw.shape[1])
            dst.createDimension("sw_band", self.bext_sw.shape[1])
            dst.createDimension("coszen", len(self.mu_vect))
            dst.createDimension("spec lw bands", len(self.T_lw_vect))
            dst.createDimension("spec sw bands", len(self.T_sw_vect))

            # ── variables ──
            v = dst.createVariable("name", np.dtype("S1"), ("name_len",))
            v[:] = str2b(self.material_name, n=name_len)

            v = dst.createVariable("spec lw T", np.dtype("float64"), ("spec lw bands",))
            v[:] = self.T_lw_vect
            v = dst.createVariable(
                "spec lw band1", np.dtype("float64"), ("spec lw bands",)
            )
            v[:] = self.wavenumber_lw_bound_vect[:-1]
            v = dst.createVariable(
                "spec lw band2", np.dtype("float64"), ("spec lw bands",)
            )
            v[:] = self.wavenumber_lw_bound_vect[1:]

            v = dst.createVariable("spec sw T", np.dtype("float64"), ("spec sw bands",))
            v[:] = self.T_sw_vect
            v = dst.createVariable(
                "spec sw band1", np.dtype("float64"), ("spec sw bands",)
            )
            v[:] = self.wavenumber_sw_bound_vect[:-1]
            v = dst.createVariable(
                "spec sw band2", np.dtype("float64"), ("spec sw bands",)
            )
            v[:] = self.wavenumber_sw_bound_vect[1:]

            v = dst.createVariable("density", np.dtype("float64"), ())
            v[:] = self.rho_mks
            dst["density"].setncatts(
                {
                    "long_name": "Combined Aerosol and Water Material Density",
                    "units": "kg m^-3",
                }
            )

            v = dst.createVariable("mu_samples", np.dtype("float64"), ("mu_samples",))
            v[:] = self.logr_vect
            dst["mu_samples"].setncatts(
                {"long_name": "ln(geometric mean radius)", "units": "ln(m)"}
            )

            v = dst.createVariable("coszen", np.dtype("float64"), ("coszen",))
            v[:] = self.mu_vect
            dst["coszen"].setncatts(
                {"long_name": "cos of zenith angle", "units": "none"}
            )

            long_name_prefix = "Volume specific"
            units_str = "m^-1"

            v = dst.createVariable(
                "babs_lw", np.dtype("float64"), ("mu_samples", "lw_band")
            )
            v[:] = self.babs_lw
            dst["babs_lw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} Absorption Coefficient",
                    "units": units_str,
                }
            )

            v = dst.createVariable(
                "bsca_sw", np.dtype("float64"), ("mu_samples", "sw_band")
            )
            v[:] = self.bsca_sw
            dst["bsca_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} Scattering Coefficient",
                    "units": units_str,
                }
            )

            v = dst.createVariable(
                "basc_sw", np.dtype("float64"), ("mu_samples", "sw_band")
            )
            v[:] = self.basc_sw
            dst["basc_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} Asymetric Scattering "
                    f"Coefficient",
                    "units": units_str,
                }
            )

            v = dst.createVariable(
                "bext_sw", np.dtype("float64"), ("mu_samples", "sw_band")
            )
            v[:] = self.bext_sw
            dst["bext_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} Extinction Coefficient",
                    "units": units_str,
                }
            )

            v = dst.createVariable(
                "ref_spec_weight_sw", np.dtype("float64"), ("sw_band",)
            )
            v[:] = self.ref_spec_weight_sw
            dst["ref_spec_weight_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} reference sw spectrum weight",
                    "units": "",
                }
            )

            v = dst.createVariable(
                "bet_bar_bsca_sw", np.dtype("float64"), ("mu_samples", "sw_band")
            )
            v[:] = self.bet_bar_bsca_sw
            dst["bet_bar_bsca_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} beta_bar times Scattering "
                    f"Coefficient",
                    "units": units_str,
                }
            )

            v = dst.createVariable(
                "bet_mu_bsca_sw",
                np.dtype("float64"),
                ("mu_samples", "sw_band", "coszen"),
            )
            v[:] = self.bet_mu_bsca_sw
            dst["bet_mu_bsca_sw"].setncatts(
                {
                    "long_name": f"{long_name_prefix} beta_mu times Scattering "
                    f"Coefficient",
                    "units": units_str,
                }
            )


# ═══════════════════════════════════════════════════
#  Configuration (CLI-populated)
# ═══════════════════════════════════════════════════


@dataclass
class OpticalConfig:
    """Resolved configuration for an optical-table run.

    Drives both pipeline stages for one material: Stage 1 writes the
    intermediate refractive-index CSV (``optical_csv``), Stage 2 reads it back
    and writes the climlab RRTMG table (``out_file``). Defaults reproduce the
    standard table-generation run.
    """

    # Stage-1 refractive-index wavelength grid (micron, log-spaced).
    ri_lambda_min: float = 0.1
    ri_lambda_max: float = 1000.0
    ri_n_lambda: int = 20001

    # Stage-2 Mie radius grid (nm) and angular sampling.
    rmin_nm: float = 50.0
    r_nm_ref: float = 250.0
    nr: int = 51
    n_theta_tag: int = 1000
    n_mu: int = 81
    mu_min: float = 0.05
    mu_max: float = 1.0
    T_sw: float = 6000.0

    # Number of worker processes for the Stage-2 radius loop. <= 1 runs the
    # serial code path; > 1 distributes radii across a ProcessPoolExecutor.
    n_workers: int = 1

    # Output paths. When left as None they are resolved next to the
    # working directory by the material module's generate().
    optical_csv: str | None = None
    out_file: str | None = None

    def radius_bounds_nm(self) -> tuple[float, float]:
        """Return ``(rmin_nm, rmax_nm)`` for the Stage-2 radius grid.

        ``rmax_nm`` is derived geometrically so that ``r_nm_ref`` sits at the
        grid midpoint, matching the source pipeline's radius spacing.
        """
        nref = self.nr // 2
        a = (self.r_nm_ref / self.rmin_nm) ** (1.0 / nref)
        rmax_nm = self.rmin_nm * a ** (self.nr - 1)
        return self.rmin_nm, rmax_nm

    def stamp_config(self) -> dict:
        """The config dict recorded in the provenance stamp."""
        rmin_nm, rmax_nm = self.radius_bounds_nm()
        return {
            "ri_lambda_min": self.ri_lambda_min,
            "ri_lambda_max": self.ri_lambda_max,
            "ri_n_lambda": self.ri_n_lambda,
            "rmin_nm": rmin_nm,
            "rmax_nm": rmax_nm,
            "r_nm_ref": self.r_nm_ref,
            "nr": self.nr,
            "n_theta_tag": self.n_theta_tag,
            "n_mu": self.n_mu,
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
            "T_sw": self.T_sw,
            "n_workers": self.n_workers,
        }


# ═══════════════════════════════════════════════════
#  Shared Stage-1 -> Stage-2 driver
# ═══════════════════════════════════════════════════


def write_refractive_index_csv(path, vlambda_vect, n_vect, k_vect):
    """Write a Stage-1 intermediate refractive-index table to ``path``.

    The CSV columns ``Wavelength (um)`` / ``Real Part`` / ``Imaginary Part``
    are the contract consumed by :class:`Particle` in Stage 2.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(
        columns=["Wavelength (um)", "Real Part", "Imaginary Part"],
        data=[[vlam, n, k] for vlam, n, k in zip(vlambda_vect, n_vect, k_vect)],
    )
    df.to_csv(path, index=False)


def build_optical_table(
    material_name: str,
    cfg: OpticalConfig,
    refractive_index_fn: Callable[[OpticalConfig], tuple],
    *,
    generator: str,
    source: str,
    processing: str,
) -> str:
    """Run the full two-stage optical pipeline for one material.

    Stage 1: ``refractive_index_fn(cfg)`` returns the material's
    ``(vlambda_vect, n_vect, k_vect)`` refractive-index spectrum, which is
    written to ``cfg.optical_csv``.

    Stage 2: the CSV is loaded into :class:`RadiationTransferTableRRTMG`, the
    Mie + RRTMG-band tables are built, and the climlab NetCDF
    (``cfg.out_file``) is written and provenance-stamped.

    Returns the output NetCDF path.
    """
    # ── Stage 1: refractive-index table ──
    vlambda_vect, n_vect, k_vect = refractive_index_fn(cfg)
    write_refractive_index_csv(cfg.optical_csv, vlambda_vect, n_vect, k_vect)

    # ── Stage 2: Mie + RRTMG-band tables ──
    rmin_nm, rmax_nm = cfg.radius_bounds_nm()

    lambda_um_sw_vect = np.exp(-np.linspace(np.log(1.0 / 0.2), np.log(1.0 / 15.0), 501))
    lambda_um_lw_vect = np.linspace(0.2, 1000.0, 20001)

    # Per-band longwave effective temperatures fitted to the RRTMG band
    # weight distribution.
    T_lw_list, _ = band_temperature_fit(
        RRTMG_LW_BAND_BOUNDS, RRTMG_LW_WEIGHT_TARGET, temp_guess=255.0 * np.ones((16,))
    )
    wavenumber_lw_bound_list = RRTMG_LW_BAND_BOUNDS[1:-1]

    rrtmg = RadiationTransferTableRRTMG(
        material_name,
        lambda_um_sw_vect,
        lambda_um_lw_vect,
        cfg.optical_csv,
        T_sw=cfg.T_sw,
        wavenumber_lw_bound_list=wavenumber_lw_bound_list,
        T_lw_list=T_lw_list,
        rmin_nm=rmin_nm,
        rmax_nm=rmax_nm,
        nr=cfg.nr,
        n_theta_tag=cfg.n_theta_tag,
        n_mu=cfg.n_mu,
        mu_min=cfg.mu_min,
        mu_max=cfg.mu_max,
        n_workers=cfg.n_workers,
    )

    # ── Build the provenance global attributes and write them into the
    #    same single NetCDF write (the table is written exactly once). ──
    stamp_attrs = provenance.stamp(
        xr.Dataset(),
        generator=generator,
        config={
            **cfg.stamp_config(),
            "optical_csv": os.path.basename(cfg.optical_csv),
            "out_file": os.path.basename(cfg.out_file),
        },
        source=source,
        processing=processing,
        period="",
        repo_dir=_REPO_DIR,
    ).attrs

    rrtmg.generate_nc_file(cfg.out_file, global_attrs=stamp_attrs)
    return cfg.out_file
