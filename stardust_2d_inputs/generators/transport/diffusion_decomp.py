#!/usr/bin/env python3
"""transport.diffusion_decomp — ERA5 diffusion + Pitari TEM decomposition generator.

Computes the eddy diffusivity tensor from ERA5 model-level data (L137) by
displacement tracking, with the Pitari Transformed-Eulerian-Mean
decomposition of the eddy transport stream function. Writes one stamped
NetCDF per ``(year, month)``.

The output is the *raw* decomposition — the diffusion tensor is reported as
time-mean values with std and std/|mean| ratios, with no postprocessing
(no floor replacement, no eigenvalue constraint, no negative masking);
downstream code chooses any cutoff criteria. Specifically:

  A) No omega smoothing for K_pp (``omega_smooth_n = 1``).
  B) No postprocessing on the diffusion tensor — raw mean values as-is.
  C) std/|mean| ratio output for every tensor component.
  D) Traditional TEM velocities and stream functions in the output.
  E) Polar caps for velocities/stream functions filled by linear
     interpolation along latitude (the diffusion tensor is left raw).

Configuration is via :class:`TemConfig` (CLI-populated); the output is
stamped via :func:`core.provenance.stamp` before writing.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid

from ...core import provenance

GENERATOR = "transport.diffusion_decomp"

# Repository root for the provenance git-state stamp.
_REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# ═══════════════════════════════════════════════════
#  Configuration (was the module-level USER CONFIGURATION block)
# ═══════════════════════════════════════════════════


@dataclass
class TemConfig:
    """Resolved configuration for a TEM-decomposition run.

    Replaces the original module-level ``USER CONFIGURATION`` globals. Field
    values and defaults are carried over unchanged from that block; they are
    populated from CLI arguments by :func:`config_from_args`.
    """

    year_start: int
    year_end: int
    month_start: int
    month_end: int
    use_gpu: bool = False
    n_workers: int = 1
    zarr_path: str | None = None
    output_dir: str = "for_transfer_ml_TEMdecomp_raw"
    p_top: float = 1.0
    p_bottom: float = 500.0
    # omega smoothing window. N=1 -> no smoothing: raw omega' used for K_pp
    # (and all other tensor components).
    omega_smooth_n: int = 1

    def jobs(self) -> list[tuple[int, int]]:
        """The ``(year, month)`` pairs this run covers."""
        return [
            (y, m)
            for y in range(self.year_start, self.year_end + 1)
            for m in range(self.month_start, self.month_end + 1)
        ]

    def stamp_config(self) -> dict:
        """The config dict recorded in the provenance stamp."""
        return {
            "year_start": self.year_start,
            "year_end": self.year_end,
            "month_start": self.month_start,
            "month_end": self.month_end,
            "use_gpu": self.use_gpu,
            "n_workers": self.n_workers,
            "zarr_path": self.zarr_path,
            "output_dir": self.output_dir,
            "p_top": self.p_top,
            "p_bottom": self.p_bottom,
            "omega_smooth_n": self.omega_smooth_n,
        }


# ── Optional CuPy (GPU) acceleration ──
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as gpu_map_coordinates

    HAS_GPU = True
    print(f"GPU available: {cp.cuda.runtime.getDeviceCount()} device(s)")
except ImportError:
    HAS_GPU = False
    print("CuPy not found — running CPU-only mode.")


# ── Physical constants (not configuration) ──
a_earth = 6.371e6
dt = 3 * 3600.0  # 3-hour timestep (s)
kappa = 0.2857  # R/cp for dry air (= 2/7)
p0_Pa = 1.0e5  # reference pressure for θ (Pa)
deg_per_rad = 180.0 / np.pi
pa_to_hpa = 1.0 / 100.0
POLAR_LAT_THRESHOLD = 85.0


# ═══════════════════════════════════════════════════
#  ERA5 L137 hybrid coefficients
# ═══════════════════════════════════════════════════
# fmt: off
A_HALF = np.array([
    0.000000, 2.000365, 3.102241, 4.666084, 6.827977, 9.746966, 13.605424, 18.608931,
    24.985718, 32.985710, 42.879242, 54.955463, 69.520576, 86.895882, 107.415741, 131.425507,
    159.279404, 191.338562, 227.968948, 269.539581, 316.420746, 368.982361, 427.592499, 492.616028,
    564.413452, 643.339905, 729.744141, 823.967834, 926.344910, 1037.201172, 1156.853638, 1285.610352,
    1423.770142, 1571.622925, 1729.448975, 1897.519287, 2076.095947, 2265.431641, 2465.770508, 2677.348145,
    2900.391357, 3135.119385, 3381.743652, 3640.468262, 3911.490479, 4194.930664, 4490.817383, 4799.149414,
    5119.895020, 5452.990723, 5798.344727, 6156.074219, 6526.946777, 6911.870605, 7311.869141, 7727.412109,
    8159.354004, 8608.525391, 9076.400391, 9562.682617, 10065.978516, 10584.631836, 11116.662109, 11660.067383,
    12211.547852, 12766.873047, 13324.668945, 13881.331055, 14432.139648, 14975.615234, 15508.256836, 16026.115234,
    16527.322266, 17008.789063, 17467.613281, 17901.621094, 18308.433594, 18685.718750, 19031.289063, 19343.511719,
    19620.042969, 19859.390625, 20059.931641, 20219.664063, 20337.863281, 20412.308594, 20442.078125, 20425.718750,
    20361.816406, 20249.511719, 20087.085938, 19874.025391, 19608.572266, 19290.226563, 18917.460938, 18489.707031,
    18006.925781, 17471.839844, 16888.687500, 16262.046875, 15596.695313, 14898.453125, 14173.324219, 13427.769531,
    12668.257813, 11901.339844, 11133.304688, 10370.175781, 9617.515625, 8880.453125, 8163.375000, 7470.343750,
    6804.421875, 6168.531250, 5564.382813, 4993.796875, 4457.375000, 3955.960938, 3489.234375, 3057.265625,
    2659.140625, 2294.242188, 1961.500000, 1659.476563, 1387.546875, 1143.250000, 926.507813, 734.992188,
    568.062500, 424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813,
    0.000000, 0.000000
], dtype=np.float64)

B_HALF = np.array([
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000007,
    0.000024, 0.000059, 0.000112, 0.000199, 0.000340, 0.000562, 0.000890, 0.001353,
    0.001992, 0.002857, 0.003971, 0.005378, 0.007133, 0.009261, 0.011806, 0.014816,
    0.018318, 0.022355, 0.026964, 0.032176, 0.038026, 0.044548, 0.051773, 0.059728,
    0.068448, 0.077958, 0.088286, 0.099462, 0.111505, 0.124448, 0.138313, 0.153125,
    0.168910, 0.185689, 0.203491, 0.222333, 0.242244, 0.263242, 0.285354, 0.308598,
    0.332939, 0.358254, 0.384363, 0.411125, 0.438391, 0.466003, 0.493800, 0.521619,
    0.549301, 0.576692, 0.603648, 0.630036, 0.655736, 0.680643, 0.704669, 0.727739,
    0.749797, 0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432, 0.873929,
    0.887408, 0.899900, 0.911448, 0.922096, 0.931881, 0.940860, 0.949064, 0.956550,
    0.963352, 0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003,
    0.997630, 1.000000,
], dtype=np.float64)
# fmt: on


def compute_full_level_pressure(lnsp_mean=None, ps_default=101325.0):
    if lnsp_mean is not None:
        ps = np.exp(lnsp_mean)
    else:
        ps = ps_default
    a_full = 0.5 * (A_HALF[:-1] + A_HALF[1:])
    b_full = 0.5 * (B_HALF[:-1] + B_HALF[1:])
    p_full = a_full + b_full * ps
    return p_full, a_full, b_full


def select_levels(p_top_hpa, p_bottom_hpa, ps_default=101325.0):
    p_full, a_full, b_full = compute_full_level_pressure(ps_default=ps_default)
    p_full_hpa = p_full / 100.0
    mask = (p_full_hpa >= p_top_hpa) & (p_full_hpa <= p_bottom_hpa)
    sel_idx = np.where(mask)[0]
    print(
        f"  Selected {len(sel_idx)} model levels between {p_top_hpa}–{p_bottom_hpa} hPa"
    )
    print(f"  Level indices: {sel_idx[0]+1}–{sel_idx[-1]+1} (1-based)")
    print(
        f"  Pressure range: {p_full_hpa[sel_idx[0]]:.2f}–{p_full_hpa[sel_idx[-1]]:.2f} hPa"
    )
    b_sel = b_full[sel_idx]
    first_significant_b = np.where(b_sel > 1e-4)[0]
    if len(first_significant_b) > 0:
        idx_b = first_significant_b[0]
        print(
            f"  Hybrid b > 1e-4 starting at level {sel_idx[idx_b]+1} "
            f"({p_full_hpa[sel_idx[idx_b]]:.1f} hPa)"
        )
    else:
        print("  All selected levels have b ≈ 0: pure pressure levels")
    return sel_idx, p_full_hpa[sel_idx], a_full[sel_idx], b_full[sel_idx]


# ═══════════════════════════════════════════════════
#  Core interpolation
# ═══════════════════════════════════════════════════


def _fractional_indices(dep_vals, grid_vals):
    if grid_vals[0] > grid_vals[-1]:
        grid_vals = grid_vals[::-1]
        idx = np.searchsorted(grid_vals, dep_vals) - 1
        idx = np.clip(idx, 0, len(grid_vals) - 2)
        frac = (dep_vals - grid_vals[idx]) / (grid_vals[idx + 1] - grid_vals[idx])
        return (len(grid_vals) - 1) - idx - frac
    else:
        idx = np.searchsorted(grid_vals, dep_vals) - 1
        idx = np.clip(idx, 0, len(grid_vals) - 2)
        frac = (dep_vals - grid_vals[idx]) / (grid_vals[idx + 1] - grid_vals[idx])
        return idx + frac


def interp_3d_cpu(field_3d, fi_lev, fi_lat, fi_lon):
    coords = np.array([fi_lev.ravel(), fi_lat.ravel(), fi_lon.ravel()])
    result = map_coordinates(field_3d, coords, order=1, mode="nearest")
    return result.reshape(field_3d.shape).astype(np.float32)


def interp_3d_gpu(field_3d, fi_lev, fi_lat, fi_lon):
    field_gpu = cp.asarray(field_3d)
    coords_gpu = cp.array(
        [
            cp.asarray(fi_lev).ravel(),
            cp.asarray(fi_lat).ravel(),
            cp.asarray(fi_lon).ravel(),
        ]
    )
    result_gpu = gpu_map_coordinates(field_gpu, coords_gpu, order=1, mode="nearest")
    result = cp.asnumpy(result_gpu).reshape(field_3d.shape).astype(np.float32)
    del field_gpu, coords_gpu, result_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return result


def interp_3d(field_3d, fi_lev, fi_lat, fi_lon, use_gpu=False):
    if use_gpu and HAS_GPU:
        return interp_3d_gpu(field_3d, fi_lev, fi_lat, fi_lon)
    return interp_3d_cpu(field_3d, fi_lev, fi_lat, fi_lon)


# ═══════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════


def wrap_longitude(dep_lon, lon_min, lon_max):
    return lon_min + (dep_lon - lon_min) % 360.0


def fill_polar_interp(field_2d, lat_arr):
    """
    Fill NaN values at polar caps by 1-D linear interpolation along the
    latitude axis (per pressure level).  Used for velocities and stream
    functions only — NOT for raw diffusion tensor output.
    """
    out = field_2d.copy().astype(np.float64)
    for k in range(out.shape[0]):
        row = out[k, :]
        if not np.any(np.isnan(row)):
            continue
        valid_mask = ~np.isnan(row)
        if valid_mask.sum() < 2:
            continue
        f = interp1d(
            lat_arr[valid_mask],
            row[valid_mask],
            kind="linear",
            bounds_error=False,
            fill_value=(row[valid_mask][-1], row[valid_mask][0]),
        )
        nan_idx = np.where(np.isnan(row))[0]
        out[k, nan_idx] = f(lat_arr[nan_idx])
    return out.astype(np.float32)


def compute_v_star(psi_2d, pres_arr_hpa):
    """v* = -dΨ/dp using PCHIP.  Input psi_2d: (nlev, nlat)."""
    interp = PchipInterpolator(pres_arr_hpa, psi_2d, axis=0)
    dpsi_dp_hpa = interp(pres_arr_hpa, 1)  # dΨ/d(hPa)
    return -dpsi_dp_hpa * pa_to_hpa  # → -dΨ/d(Pa)  [m/s]


def compute_w_star(psi_2d, lat_arr, cos_phi_2d):
    """w* = (1 / a cosφ) ∂(cosφ Ψ)/∂φ.  Returns (nlev, nlat)."""
    cos_psi = cos_phi_2d * psi_2d
    dcosPsi_dphi = np.gradient(cos_psi, np.deg2rad(lat_arr), axis=1)
    return dcosPsi_dphi / (a_earth * cos_phi_2d)


def compute_sigma(theta_bar_mean, pres_arr_hpa):
    """
    Static stability σ = dθ̄/dp  (K/Pa).

    theta_bar_mean: (nlev, nlat) time-averaged zonal-mean potential temperature.
    pres_arr_hpa:   (nlev,) pressure in hPa (monotonic, low→high or high→low).

    Returns sigma: (nlev, nlat) in K/Pa.
    Note: σ < 0 everywhere in a stably stratified atmosphere
          (θ increases upward = decreasing pressure).
    """
    nlev, nlat = theta_bar_mean.shape
    sigma = np.zeros_like(theta_bar_mean)
    for j in range(nlat):
        interp = PchipInterpolator(pres_arr_hpa, theta_bar_mean[:, j])
        sigma[:, j] = interp(pres_arr_hpa, 1) / 100.0
    return sigma


def mask_polar(field_2d, lat_arr, threshold=POLAR_LAT_THRESHOLD):
    mask = np.abs(lat_arr) > threshold
    field_2d[:, mask] = np.nan
    return field_2d


# ═══════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════

ZARR_PATHS_MODEL = [
    "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1",
    "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v2",
]

# ERA5 model-level temperature is stored as 't' in the ARCO zarr
VAR_NAMES = {
    "u": ["u_component_of_wind", "u", "U"],
    "v": ["v_component_of_wind", "v", "V"],
    "w": ["vertical_velocity", "w", "omega", "W"],
    "t": ["temperature", "t", "T"],
    "lnsp": ["logarithm_of_surface_pressure", "lnsp", "LNSP"],
}


def detect_variables(ds):
    available = list(ds.data_vars) + list(ds.coords)
    detected = {}
    for key, candidates in VAR_NAMES.items():
        for name in candidates:
            if name in available:
                detected[key] = name
                break
    print(f"  Available variables: {list(ds.data_vars)}")
    print(f"  Detected mapping: {detected}")
    missing = set(VAR_NAMES.keys()) - set(detected.keys())
    if missing - {"lnsp"}:  # lnsp is optional
        print(f"  WARNING: Could not find variables for: {missing - {'lnsp'}}")
    if "t" not in detected:
        raise RuntimeError(
            "Temperature ('t') not found in dataset. "
            "Cannot compute TEM stream function without it. "
            f"Available: {available[:20]}"
        )
    return detected


def open_model_level_zarr(zarr_path=None):
    paths_to_try = [zarr_path] if zarr_path else ZARR_PATHS_MODEL
    for path in paths_to_try:
        try:
            print(f"  Trying: {path}")
            ds = xr.open_zarr(path, consolidated=True)
            print(f"  SUCCESS: Opened {path}")
            print(f"  Dimensions: {dict(ds.dims)}")
            return ds, path
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
    raise RuntimeError(
        "Could not open any model-level Zarr store. "
        "Check network connectivity and GCS access."
    )


# ═══════════════════════════════════════════════════
#  Main processing for a single month
# ═══════════════════════════════════════════════════


def process_month(year, month, cfg: TemConfig):
    """Process a single ``(year, month)`` and write its stamped NetCDF.

    Configuration (``P_TOP``, ``P_BOTTOM``, ``OMEGA_SMOOTH_N``, ``zarr_path``,
    ``output_dir``, ``use_gpu``) is taken from ``cfg``; the output dataset is
    stamped before ``to_netcdf``.
    """
    # Local bindings — the numeric body references these as plain names.
    P_TOP = cfg.p_top
    P_BOTTOM = cfg.p_bottom
    OMEGA_SMOOTH_N = cfg.omega_smooth_n
    zarr_path = cfg.zarr_path
    output_dir = cfg.output_dir
    use_gpu = cfg.use_gpu

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Processing {year}-{month:02d} (MODEL LEVELS + TEM DECOMP — RAW)")
    print(f"{'='*60}")

    filename = os.path.join(
        output_dir, f"diffusion_ml_TEMdecomp_raw_{year}_{month:02d}.nc"
    )
    if os.path.exists(filename):
        print("  File exists, skipping.")
        return

    # ── Select model levels ──
    sel_idx, pres_ref_hpa, a_coeff, b_coeff = select_levels(P_TOP, P_BOTTOM)
    nlev_sel = len(sel_idx)
    level_numbers = sel_idx + 1  # 1-based ERA5 model level numbers

    # ── Open dataset ──
    print("  Opening ERA5 model-level data...")
    ds, actual_path = open_model_level_zarr(zarr_path)
    var_map = detect_variables(ds)

    for key in ["u", "v", "w", "t"]:
        if key not in var_map:
            raise RuntimeError(
                f"Cannot proceed without '{key}'. Available: {list(ds.data_vars)}"
            )

    # ── Select time range ──
    ds = ds.sel(time=(ds.time.dt.year == year) & (ds.time.dt.month == month))

    # ── Vertical level dimension ──
    level_dim = None
    for candidate in ["level", "model_level", "hybrid"]:
        if candidate in ds.dims:
            level_dim = candidate
            break
    if level_dim is None:
        raise RuntimeError(f"Cannot find level dimension. Dims: {dict(ds.dims)}")
    print(f"  Vertical dimension: '{level_dim}' with {ds.dims[level_dim]} levels")
    ds = ds.sel({level_dim: level_numbers})

    # ── Select variables (now includes T) ──
    vars_to_load = [var_map[k] for k in ["u", "v", "w", "t"] if k in var_map]
    if "lnsp" in var_map:
        vars_to_load.append(var_map["lnsp"])
    ds = ds[vars_to_load]

    # ── Subsample ──
    ds = ds.isel(
        time=slice(0, None, 3), longitude=slice(0, None, 4), latitude=slice(0, None, 4)
    )
    ds = ds.chunk({"time": 1, level_dim: nlev_sel, "latitude": -1, "longitude": -1})

    # ── Extract coordinates ──
    lat_arr = ds.latitude.values.astype(np.float64)
    lon_arr = ds.longitude.values.astype(np.float64)
    times = ds.time.values
    n_times = len(times)
    n_steps = n_times - 1
    nlat, nlon = len(lat_arr), len(lon_arr)

    print(f"  Pressure reference: {pres_ref_hpa[0]:.2f} – {pres_ref_hpa[-1]:.2f} hPa")
    print(f"  Grid: {nlev_sel} levels × {nlat} lat × {nlon} lon, {n_times} timesteps")
    print(f"  Using {'GPU' if (use_gpu and HAS_GPU) else 'CPU'} interpolation")

    # ── Pressure in Pa for θ computation ──
    pres_3d_Pa = (pres_ref_hpa[:, np.newaxis, np.newaxis] * 100.0).astype(np.float32)

    # ── Batch loader ──
    BATCH_STEPS = 48

    def _load_batch(start_i, end_i):
        sl = slice(start_i, end_i)
        u_b = ds[var_map["u"]].isel(time=sl).values.astype(np.float32)
        v_b = ds[var_map["v"]].isel(time=sl).values.astype(np.float32)
        w_b = ds[var_map["w"]].isel(time=sl).values.astype(np.float32)
        T_b = ds[var_map["t"]].isel(time=sl).values.astype(np.float32)
        return u_b, v_b, w_b, T_b

    # ── Latitude-dependent factors ──
    cos_phi = np.cos(np.deg2rad(lat_arr))[np.newaxis, :, np.newaxis]  # (1,nlat,1)
    cos_phi_2d = np.cos(np.deg2rad(lat_arr))[np.newaxis, :]  # (1,nlat)

    # ── Load first batch ──
    b_start = 0
    b_end = min(BATCH_STEPS + 1, n_times)
    print(f"  Fetching initial batch ({b_end} timesteps) from ERA5...")
    u_bat, v_bat, w_bat, T_bat = _load_batch(b_start, b_end)

    u_prev = u_bat[0]
    v_prev = v_bat[0]
    w_prev = w_bat[0]

    vp_prev = v_prev - v_prev.mean(axis=-1, keepdims=True)
    wp_prev = w_prev - w_prev.mean(axis=-1, keepdims=True)

    # ── Displacement fields ──
    eta = np.zeros((nlev_sel, nlat, nlon), dtype=np.float32)  # meridional (m)
    xi = np.zeros((nlev_sel, nlat, nlon), dtype=np.float32)  # vertical (Pa·s)

    # ── Running sums — tensor (1st moment) ──
    K_yy_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_yp_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_py_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_pp_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)

    # ── Running sums — tensor (2nd moment for variance) ──
    K_yy_sq_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_yp_sq_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_py_sq_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    K_pp_sq_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)

    # ── Running sums — TEM quantities ──
    vT_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)
    theta_bar_sum = np.zeros((nlev_sel, nlat), dtype=np.float64)

    # ── 3-D coordinate grids ──
    pres_3d = pres_ref_hpa[:, np.newaxis, np.newaxis] * np.ones((1, nlat, nlon))
    lat_3d = lat_arr[np.newaxis, :, np.newaxis] * np.ones((nlev_sel, 1, nlon))
    lon_3d = lon_arr[np.newaxis, np.newaxis, :] * np.ones((nlev_sel, nlat, 1))

    # ── ω ring buffer ──
    wp_smooth_buf = []

    # ── Time loop ──
    print("  Advecting displacements...")
    lap_timer = time.time()
    current_day = -1

    for ti in range(1, n_times):
        # ── Backward-Euler departure coordinates ──
        dep_lon = lon_3d - (u_prev * dt / (a_earth * cos_phi)) * deg_per_rad
        dep_lat = lat_3d - (v_prev * dt / a_earth) * deg_per_rad
        dep_pres = pres_3d - (w_prev * dt) * pa_to_hpa

        dep_lon = wrap_longitude(dep_lon, float(lon_arr.min()), float(lon_arr.max()))
        dep_lat = np.clip(dep_lat, float(lat_arr.min()), float(lat_arr.max()))
        dep_pres = np.clip(
            dep_pres, float(pres_ref_hpa.min()), float(pres_ref_hpa.max())
        )

        fi_lon = _fractional_indices(dep_lon, lon_arr)
        fi_lat = _fractional_indices(dep_lat, lat_arr)
        fi_lev = _fractional_indices(dep_pres, pres_ref_hpa)

        # Interpolate to departure points
        eta_dep = interp_3d(eta, fi_lev, fi_lat, fi_lon, use_gpu)
        xi_dep = interp_3d(xi, fi_lev, fi_lat, fi_lon, use_gpu)
        vp_dep = interp_3d(vp_prev, fi_lev, fi_lat, fi_lon, use_gpu)
        wp_dep = interp_3d(wp_prev, fi_lev, fi_lat, fi_lon, use_gpu)

        # Update displacements
        eta = eta_dep + vp_dep * dt
        xi = xi_dep + wp_dep * dt

        # ── Fetch current timestep ──
        local_i = ti - b_start
        if local_i >= (b_end - b_start):
            b_start = ti
            b_end = min(ti + BATCH_STEPS, n_times)
            print(f"    Fetching next batch (steps {b_start}–{b_end})...")
            u_bat, v_bat, w_bat, T_bat = _load_batch(b_start, b_end)
            local_i = 0

        u_now = u_bat[local_i]
        v_now = v_bat[local_i]
        w_now = w_bat[local_i]
        T_now = T_bat[local_i]

        vp_now = v_now - v_now.mean(axis=-1, keepdims=True)
        wp_now = w_now - w_now.mean(axis=-1, keepdims=True)

        # ── ω smoothing (K_pp only) ──
        wp_smooth_buf.append(wp_now)
        if len(wp_smooth_buf) > OMEGA_SMOOTH_N:
            wp_smooth_buf.pop(0)
        wp_smooth_now = np.mean(wp_smooth_buf, axis=0).astype(np.float32)

        # ── Tensor components (per timestep, zonal mean) ──
        K_yy_t = (vp_now * eta).mean(axis=-1)  # (nlev, nlat)
        K_yp_t = (vp_now * xi).mean(axis=-1)
        K_py_t = (wp_now * eta).mean(axis=-1)
        K_pp_t = (wp_smooth_now * xi).mean(axis=-1)

        # 1st moment accumulation
        K_yy_sum += K_yy_t
        K_yp_sum += K_yp_t
        K_py_sum += K_py_t
        K_pp_sum += K_pp_t

        # 2nd moment accumulation (for variance)
        K_yy_sq_sum += K_yy_t**2
        K_yp_sq_sum += K_yp_t**2
        K_py_sq_sum += K_py_t**2
        K_pp_sq_sum += K_pp_t**2

        # ── TEM accumulators ──
        theta_now = T_now * (p0_Pa / pres_3d_Pa) ** kappa
        theta_bar_now = theta_now.mean(axis=-1)
        theta_prime = theta_now - theta_bar_now[:, :, np.newaxis]

        vT_sum += (vp_now * theta_prime).mean(axis=-1)
        theta_bar_sum += theta_bar_now

        # ── Advance ──
        u_prev, v_prev, w_prev = u_now, v_now, w_now
        vp_prev, wp_prev = vp_now, wp_now

        # ── Progress ──
        t_stamp = times[ti]
        day = (
            int(
                np.datetime64(t_stamp, "D").astype("datetime64[D]").astype(int)
                - np.datetime64(f"{year}-{month:02d}-01")
                .astype("datetime64[D]")
                .astype(int)
            )
            + 1
        )
        if day != current_day:
            current_day = day
            elapsed = time.time() - lap_timer
            rate = ti / elapsed if elapsed > 0 else 0
            eta_remaining = (n_steps - ti) / rate if rate > 0 else 0
            print(
                f"    Day {day:2d} | Step {ti}/{n_steps}  "
                f"({100*ti/n_steps:.0f}%)  "
                f"{rate:.1f} steps/sec  "
                f"ETA: {eta_remaining/60:.0f} min"
            )

    # ══════════════════════════════════════════════════
    #  Post time-loop: raw tensor + SNR + TEM decomposition
    # ══════════════════════════════════════════════════
    print("  Computing raw tensor, SNR ratios, and TEM decomposition...")

    # ── Raw time-averaged tensor (NO postprocessing) ──
    K_yy_mean = (K_yy_sum / n_steps).astype(np.float32)
    K_yp_mean = (K_yp_sum / n_steps).astype(np.float32)
    K_py_mean = (K_py_sum / n_steps).astype(np.float32)
    K_pp_mean = (K_pp_sum / n_steps).astype(np.float32)

    # Symmetric and antisymmetric parts (raw, no clipping)
    D_yy_raw = K_yy_mean.copy()
    D_pp_raw = K_pp_mean.copy()
    D_yp_sym_raw = (0.5 * (K_yp_mean + K_py_mean)).astype(np.float32)
    D_yp_asym_raw = (0.5 * (K_yp_mean - K_py_mean)).astype(np.float32)  # = -Psi_displ

    # ── Variance and std/|mean| ratio for each K component ──
    def compute_snr(K_sum, K_sq_sum, n):
        """Compute std/|mean| ratio from running sums."""
        mean = K_sum / n
        var = K_sq_sum / n - mean**2
        # Protect against numerical noise giving tiny negative variance
        std = np.sqrt(np.maximum(var, 0.0))
        # std/|mean| ratio; where mean≈0 the ratio is meaningless → set to inf
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(mean) > 0, std / np.abs(mean), np.inf)
        return std.astype(np.float32), ratio.astype(np.float32)

    K_yy_std, K_yy_snr = compute_snr(K_yy_sum, K_yy_sq_sum, n_steps)
    K_yp_std, K_yp_snr = compute_snr(K_yp_sum, K_yp_sq_sum, n_steps)
    K_py_std, K_py_snr = compute_snr(K_py_sum, K_py_sq_sum, n_steps)
    K_pp_std, K_pp_snr = compute_snr(K_pp_sum, K_pp_sq_sum, n_steps)

    # Also compute SNR for the symmetrized off-diagonal D_phi_p
    # D_phi_p_t = 0.5*(K_yp_t + K_py_t) for each timestep
    # mean(D_phi_p) = D_yp_sym_raw
    # var(D_phi_p) = var(0.5*(K_yp + K_py)) = 0.25*(var(K_yp) + var(K_py) + 2*cov(K_yp,K_py))
    # We don't have the cross term, so approximate from the K_yp and K_py individual SNR.
    # Better: just report K_yp and K_py individually and let user combine.

    # ── Diagnostics ──
    n_Dpp_neg = int(np.sum(K_pp_mean < 0))
    n_Dyy_neg = int(np.sum(K_yy_mean < 0))
    print(
        f"  K_pp < 0 (raw mean): {n_Dpp_neg}/{K_pp_mean.size} "
        f"({100*n_Dpp_neg/K_pp_mean.size:.1f}%)"
    )
    print(
        f"  K_yy < 0 (raw mean): {n_Dyy_neg}/{K_yy_mean.size} "
        f"({100*n_Dyy_neg/K_yy_mean.size:.1f}%)"
    )

    # Fraction with SNR < 1.5
    finite_pp = np.isfinite(K_pp_snr)
    finite_yy = np.isfinite(K_yy_snr)
    n_good_pp = int(np.sum(finite_pp & (K_pp_snr < 1.5) & (K_pp_mean > 0)))
    n_good_yy = int(np.sum(finite_yy & (K_yy_snr < 1.5) & (K_yy_mean > 0)))
    print(
        f"  K_pp physical (>0 AND snr<1.5): {n_good_pp}/{K_pp_mean.size} "
        f"({100*n_good_pp/K_pp_mean.size:.1f}%)"
    )
    print(
        f"  K_yy physical (>0 AND snr<1.5): {n_good_yy}/{K_yy_mean.size} "
        f"({100*n_good_yy/K_yy_mean.size:.1f}%)"
    )

    # ── Displacement stream function (raw) ──
    Psi_displ = -D_yp_asym_raw  # = -0.5*(K_yp - K_py)

    # ── TEM quantities ──
    vT_bar = (vT_sum / n_steps).astype(np.float64)
    theta_bar = (theta_bar_sum / n_steps).astype(np.float64)

    sigma = compute_sigma(theta_bar, pres_ref_hpa)
    sigma_safe = np.where(np.abs(sigma) < 1e-6, np.sign(sigma) * 1e-6, sigma)

    Psi_TEM = (vT_bar / sigma_safe).astype(np.float32)
    Psi_corr = (Psi_displ - Psi_TEM).astype(np.float32)

    ratio_corr_TEM = (np.abs(Psi_corr) / (np.abs(Psi_TEM) + 1e-12)).astype(np.float32)

    print(
        f"  σ range: {sigma.min():.2e} – {sigma.max():.2e} K/Pa  "
        f"(expect negative everywhere)"
    )
    print(f"  Psi_TEM  max abs: {np.max(np.abs(Psi_TEM)):.2e} m Pa/s")
    print(f"  Psi_displ max abs: {np.max(np.abs(Psi_displ)):.2e} m Pa/s")
    print(f"  Psi_corr max abs: {np.max(np.abs(Psi_corr)):.2e} m Pa/s")
    print("  median |ratio| (25–75°N, 10–100 hPa): ", end="")
    lat_mid = (np.abs(lat_arr) >= 25) & (np.abs(lat_arr) <= 75)
    lev_strat = (pres_ref_hpa >= 10) & (pres_ref_hpa <= 100)
    if lat_mid.any() and lev_strat.any():
        r_sub = ratio_corr_TEM[np.ix_(lev_strat, lat_mid)]
        print(f"{np.nanmedian(r_sub):.3f}")
    else:
        print("(grid too coarse to select)")

    # ── Polar Ψ_corr treatment ──
    polar_mask = np.abs(lat_arr) > POLAR_LAT_THRESHOLD
    Psi_corr[:, polar_mask] = 0.0

    # ── Residual velocities from each stream function component ──
    v_star_TEM = compute_v_star(Psi_TEM.astype(np.float64), pres_ref_hpa).astype(
        np.float32
    )
    w_star_TEM = compute_w_star(Psi_TEM.astype(np.float64), lat_arr, cos_phi_2d).astype(
        np.float32
    )

    v_star_corr = compute_v_star(Psi_corr.astype(np.float64), pres_ref_hpa).astype(
        np.float32
    )
    w_star_corr = compute_w_star(
        Psi_corr.astype(np.float64), lat_arr, cos_phi_2d
    ).astype(np.float32)

    v_mean_all = (
        ds[var_map["v"]].mean(dim=["longitude", "time"]).values.astype(np.float32)
    )
    w_mean_all = (
        ds[var_map["w"]].mean(dim=["longitude", "time"]).values.astype(np.float32)
    )

    v_total = (v_mean_all + v_star_TEM + v_star_corr).astype(np.float32)
    w_total = (w_mean_all + w_star_TEM + w_star_corr).astype(np.float32)

    # ── Stream function diagnostics ──
    pres_Pa = pres_ref_hpa * 100.0

    Psi_mean = (
        -cumulative_trapezoid(v_mean_all.astype(np.float64), pres_Pa, axis=0, initial=0)
    ).astype(np.float32)

    Psi_res_TEM = (
        -cumulative_trapezoid(
            (v_mean_all + v_star_TEM).astype(np.float64), pres_Pa, axis=0, initial=0
        )
    ).astype(np.float32)

    Psi_total_arr = (
        -cumulative_trapezoid(v_total.astype(np.float64), pres_Pa, axis=0, initial=0)
    ).astype(np.float32)

    # ── Fill polar caps for velocities/stream functions (NOT for raw tensor) ──
    Psi_mean = fill_polar_interp(Psi_mean, lat_arr)
    Psi_res_TEM = fill_polar_interp(Psi_res_TEM, lat_arr)
    Psi_total_arr = fill_polar_interp(Psi_total_arr, lat_arr)

    Psi_displ_filled = fill_polar_interp(Psi_displ.copy(), lat_arr)

    v_star_TEM = fill_polar_interp(v_star_TEM, lat_arr)
    w_star_TEM = fill_polar_interp(w_star_TEM, lat_arr)
    v_star_corr = fill_polar_interp(v_star_corr, lat_arr)
    w_star_corr = fill_polar_interp(w_star_corr, lat_arr)
    v_mean_all = fill_polar_interp(v_mean_all, lat_arr)
    w_mean_all = fill_polar_interp(w_mean_all, lat_arr)

    v_total = (v_mean_all + v_star_TEM + v_star_corr).astype(np.float32)
    w_total = (w_mean_all + w_star_TEM + w_star_corr).astype(np.float32)

    # ── Save to NetCDF ──
    res = xr.Dataset(
        {
            # ════════════════════════════════════════════════
            #  RAW TENSOR — all 4 elements, no postprocessing
            # ════════════════════════════════════════════════
            "K_phi_phi": (
                ("level", "latitude"),
                K_yy_mean,
                {
                    "units": "m2 s-1",
                    "long_name": "Raw mean K_phi_phi (= D_phi_phi before any processing)",
                },
            ),
            "K_phi_p": (
                ("level", "latitude"),
                K_yp_mean,
                {
                    "units": "m Pa s-1",
                    "long_name": "Raw mean K_phi_p (v' * xi, zonal mean)",
                },
            ),
            "K_p_phi": (
                ("level", "latitude"),
                K_py_mean,
                {
                    "units": "m Pa s-1",
                    "long_name": "Raw mean K_p_phi (omega' * eta, zonal mean)",
                },
            ),
            "K_pp": (
                ("level", "latitude"),
                K_pp_mean,
                {
                    "units": "Pa2 s-1",
                    "long_name": "Raw mean K_pp (= D_pp before any processing)",
                },
            ),
            # ── Symmetric / antisymmetric decomposition (derived, for convenience) ──
            "D_phi_p_sym": (
                ("level", "latitude"),
                D_yp_sym_raw,
                {
                    "units": "m Pa s-1",
                    "long_name": "Symmetric off-diagonal: 0.5*(K_phi_p + K_p_phi)",
                },
            ),
            "D_phi_p_asym": (
                ("level", "latitude"),
                D_yp_asym_raw,
                {
                    "units": "m Pa s-1",
                    "long_name": "Antisymmetric off-diagonal: 0.5*(K_phi_p - K_p_phi) "
                    "(= -Psi_displ)",
                },
            ),
            # ════════════════════════════════════════════════
            #  STD and STD/|MEAN| ratios for each K element
            # ════════════════════════════════════════════════
            "K_phi_phi_std": (
                ("level", "latitude"),
                K_yy_std,
                {
                    "units": "m2 s-1",
                    "long_name": "Temporal std of K_phi_phi across timesteps",
                },
            ),
            "K_phi_p_std": (
                ("level", "latitude"),
                K_yp_std,
                {
                    "units": "m Pa s-1",
                    "long_name": "Temporal std of K_phi_p across timesteps",
                },
            ),
            "K_p_phi_std": (
                ("level", "latitude"),
                K_py_std,
                {
                    "units": "m Pa s-1",
                    "long_name": "Temporal std of K_p_phi across timesteps",
                },
            ),
            "K_pp_std": (
                ("level", "latitude"),
                K_pp_std,
                {
                    "units": "Pa2 s-1",
                    "long_name": "Temporal std of K_pp across timesteps",
                },
            ),
            "K_phi_phi_snr": (
                ("level", "latitude"),
                K_yy_snr,
                {
                    "units": "1",
                    "long_name": "std/|mean| ratio for K_phi_phi " "(inf where mean=0)",
                },
            ),
            "K_phi_p_snr": (
                ("level", "latitude"),
                K_yp_snr,
                {
                    "units": "1",
                    "long_name": "std/|mean| ratio for K_phi_p " "(inf where mean=0)",
                },
            ),
            "K_p_phi_snr": (
                ("level", "latitude"),
                K_py_snr,
                {
                    "units": "1",
                    "long_name": "std/|mean| ratio for K_p_phi " "(inf where mean=0)",
                },
            ),
            "K_pp_snr": (
                ("level", "latitude"),
                K_pp_snr,
                {
                    "units": "1",
                    "long_name": "std/|mean| ratio for K_pp " "(inf where mean=0)",
                },
            ),
            # ════════════════════════════════════════════════
            #  STREAM FUNCTIONS
            # ════════════════════════════════════════════════
            "Psi_mean": (
                ("level", "latitude"),
                Psi_mean,
                {
                    "units": "m Pa s-1",
                    "long_name": "Eulerian mean stream function -int(v_bar dp)",
                },
            ),
            "Psi_res_TEM": (
                ("level", "latitude"),
                Psi_res_TEM,
                {
                    "units": "m Pa s-1",
                    "long_name": "TEM residual stream function "
                    "-int((v_bar + v_star_TEM) dp)",
                },
            ),
            "Psi_total": (
                ("level", "latitude"),
                Psi_total_arr,
                {
                    "units": "m Pa s-1",
                    "long_name": "Total stream function -int(v_total dp)",
                },
            ),
            # ── TEM / displacement diagnostics ──
            "vT_bar": (
                ("level", "latitude"),
                vT_bar.astype(np.float32),
                {
                    "units": "m K s-1",
                    "long_name": "Eddy heat flux v'theta' (time+zonal mean)",
                },
            ),
            "sigma": (
                ("level", "latitude"),
                sigma.astype(np.float32),
                {
                    "units": "K Pa-1",
                    "long_name": "Static stability dtheta_bar/dp (negative=stable)",
                },
            ),
            "ratio_corr_TEM": (
                ("level", "latitude"),
                ratio_corr_TEM,
                {
                    "units": "1",
                    "long_name": "|Psi_corr|/|Psi_TEM| — QG validity diagnostic",
                },
            ),
            # ════════════════════════════════════════════════
            #  VELOCITY DECOMPOSITION (polar-filled, NaN-free)
            # ════════════════════════════════════════════════
            "v_mean": (
                ("level", "latitude"),
                v_mean_all,
                {"units": "m s-1", "long_name": "Eulerian mean meridional velocity"},
            ),
            "v_star_TEM": (
                ("level", "latitude"),
                v_star_TEM,
                {
                    "units": "m s-1",
                    "long_name": "TEM eddy residual meridional velocity",
                },
            ),
            "v_star_corr": (
                ("level", "latitude"),
                v_star_corr,
                {
                    "units": "m s-1",
                    "long_name": "Correction eddy residual meridional velocity",
                },
            ),
            "v_total": (
                ("level", "latitude"),
                v_total,
                {"units": "m s-1", "long_name": "Total meridional velocity"},
            ),
            "w_mean": (
                ("level", "latitude"),
                w_mean_all,
                {"units": "Pa s-1", "long_name": "Eulerian mean vertical velocity"},
            ),
            "w_star_TEM": (
                ("level", "latitude"),
                w_star_TEM,
                {"units": "Pa s-1", "long_name": "TEM eddy residual vertical velocity"},
            ),
            "w_star_corr": (
                ("level", "latitude"),
                w_star_corr,
                {
                    "units": "Pa s-1",
                    "long_name": "Correction eddy residual vertical velocity",
                },
            ),
            "w_total": (
                ("level", "latitude"),
                w_total,
                {"units": "Pa s-1", "long_name": "Total vertical velocity"},
            ),
        },
        coords={
            "year": [year],
            "month": [month],
            "level": pres_ref_hpa,
            "latitude": lat_arr,
        },
    )

    res.attrs["vertical_coordinate"] = "pressure_hPa_from_hybrid_coefficients"
    res.attrs["level_indices_1based"] = f"{level_numbers[0]}-{level_numbers[-1]}"
    res.attrs["p_top_hPa"] = float(pres_ref_hpa[0])
    res.attrs["p_bottom_hPa"] = float(pres_ref_hpa[-1])
    res.attrs["source"] = "ERA5 model levels (L137)"
    res.attrs["n_timesteps"] = n_steps
    res.attrs["decomposition"] = (
        "TEM decomposition; the diffusion tensor is raw (no postprocessing). "
        "All K tensor elements are raw means with std and std/|mean| ratios. "
        "No floor replacement, no eigenvalue constraint, no negative masking. "
        "User decides cutoff criteria downstream. "
        "Stream functions: Psi_mean=-int(v_mean dp), "
        "Psi_res_TEM=-int((v_mean+v_star_TEM) dp), "
        "Psi_total=-int(v_total dp). "
        "Psi_corr zero at |lat|>85 deg. "
        "No omega smoothing (OMEGA_SMOOTH_N=1). "
        "Velocities and stream functions polar-filled (NaN-free). "
        "Diffusion tensor NOT polar-filled (raw values at all latitudes)."
    )
    res.attrs["omega_smooth_n"] = OMEGA_SMOOTH_N
    res.attrs["snr_note"] = (
        "K_*_snr = std/|mean| across timesteps. "
        "inf where mean=0. "
        "Low values (< ~1.5) indicate robust signal. "
        "User should combine with sign check (positive for diagonal) for masking."
    )

    # ── Provenance stamp (scaffolding addition — does not touch the data) ──
    provenance.stamp(
        res,
        generator=GENERATOR,
        config={**cfg.stamp_config(), "year": year, "month": month},
        source="ERA5 model levels (L137), ARCO-ERA5 GCS Zarr",
        processing=(
            "Pitari TEM decomposition of eddy transport stream function; "
            "displacement-tracking K-tensor; RAW (no tensor postprocessing)"
        ),
        period=f"{year}-{month:02d}",
        repo_dir=_REPO_DIR,
    )

    res.to_netcdf(filename)
    elapsed_total = time.time() - t_start
    print(f"  Saved: {filename}")
    print(f"  Total time for {year}-{month:02d}: {elapsed_total/60:.1f} min")
    return elapsed_total


# ═══════════════════════════════════════════════════
#  CLI / orchestration
# ═══════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ERA5 diffusion + TEM decomposition — RAW output (model levels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs raw K tensor means + std/|mean| ratios.
No floor replacement, no eigenvalue constraint, no negative masking.

Examples:
  Single month test:
      python -m stardust_2d_inputs.generators.transport.diffusion_decomp --years 2010 2010 --months 1 1

  Full year:
      python -m stardust_2d_inputs.generators.transport.diffusion_decomp --years 2010 2010 --months 1 12

  Multi-year:
      python -m stardust_2d_inputs.generators.transport.diffusion_decomp --years 2008 2017 --months 1 12
""",
    )
    parser.add_argument(
        "--years", type=int, nargs=2, required=True, metavar=("START", "END")
    )
    parser.add_argument(
        "--months", type=int, nargs=2, required=True, metavar=("START", "END")
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--zarr", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="for_transfer_ml_TEMdecomp_raw")
    parser.add_argument(
        "--p-top",
        type=float,
        default=1.0,
        help="top pressure of the level window, hPa (default: %(default)s)",
    )
    parser.add_argument(
        "--p-bottom",
        type=float,
        default=500.0,
        help="bottom pressure of the level window, hPa (default: %(default)s)",
    )
    parser.add_argument(
        "--omega-smooth-n",
        type=int,
        default=1,
        help="omega smoothing window; 1 = no smoothing (RAW; default: %(default)s)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TemConfig:
    """Build a :class:`TemConfig` from parsed CLI arguments."""
    year_start, year_end = args.years
    month_start, month_end = args.months
    return TemConfig(
        year_start=year_start,
        year_end=year_end,
        month_start=month_start,
        month_end=month_end,
        use_gpu=args.gpu,
        n_workers=args.workers,
        zarr_path=args.zarr,
        output_dir=args.outdir,
        p_top=args.p_top,
        p_bottom=args.p_bottom,
        omega_smooth_n=args.omega_smooth_n,
    )


def run(cfg: TemConfig) -> None:
    """Run the TEM decomposition for every ``(year, month)`` in ``cfg``."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL-LEVEL DIFFUSION + TEM DECOMPOSITION — RAW OUTPUT")
    print("=" * 60)
    sel_idx, pres_ref, _, _ = select_levels(cfg.p_top, cfg.p_bottom)
    print(f"Years:  {cfg.year_start}–{cfg.year_end}")
    print(f"Months: {cfg.month_start}–{cfg.month_end}")
    print()

    jobs = cfg.jobs()

    if cfg.n_workers == 1:
        for year, month in jobs:
            process_month(year, month, cfg)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            futures = [ex.submit(process_month, y, m, cfg) for (y, m) in jobs]
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    print(f"  ERROR in worker: {e}")


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    run(config_from_args(args))


if __name__ == "__main__":
    main()
