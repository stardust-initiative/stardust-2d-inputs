#!/usr/bin/env python3
"""ERA5 isentropic eddy-diffusion tensor: fast operational version (monthly means).

Same numerics as era5_monthly_diffusion_81.py --method isentropic --step-minutes 30:
hourly ERA5 model levels 15-95 on 0.5 deg latitude x 1 deg longitude, theta 300-1000 K
every 5 K, 30-minute integration with Q diagnosed on hourly trajectory endpoints,
backward trapezoidal predictor/corrector trajectories (2 iterations) through physical
pressure, periodic trilinear isentropic interpolation with numerator/support transport,
mass-weighted zonal tensor, and both pressure mappings (mean-first and transform-first)
with every geometric piece. The retained outputs are bitwise identical to the original;
check this on your own machine with --mode validate, which runs the original side by side.

Removed because the tensor does not need them: the model-level method, geopotential and
height diagnostics, the H/L/S decomposition and its round-off checks, temporal-variability
statistics, daily summaries and paired statistics.

Performance changes only (no numerical change):
  * each trajectory solve + displacement update is one fused Numba pass per point
  * theta -> pressure mapping vectorized (replaces 47 Python double loops per hour)
  * remaining NumPy reductions run in theta slabs on a thread pool (row-wise, exact;
    verified against the serial result on the first hour of every run)
  * direct parallel Zarr chunk reader with background prefetch; 4 fields instead of 5
  * small uncompressed checkpoints once per simulated day
Defaults are tuned for a GCP c3-standard-8 (8 vCPU, 32 GB).

Examples
  python era5_iso_fast.py --mode year --year 2010 --output ./diffusion_2010_iso_fast
      all 12 months in sequence -> ./diffusion_2010_iso_fast/diffusion_iso_2010_MM.nc
      (skips months whose file exists, resumes an interrupted month, retries after a crash;
       rerun the same command after a reboot)
  python era5_iso_fast.py --mode run --start 2010-03-01T00 --end 2010-04-01T00 --output ./work_2010_03
      one interval -> ./work_2010_03/diffusion_iso_2010_03.nc
  python era5_iso_fast.py --mode validate --reference era5_monthly_diffusion_81.py --output validation
Output variables keep the original names (iso_mean_first_*, iso_transform_first_*, native_*, ...).
Requires numpy, numba, numcodecs, xarray, h5netcdf. --mode validate also imports the original
script (and its dask/gcsfs/zarr reader for the optional reader cross-check).
"""
from pathlib import Path
import argparse, base64, hashlib, json, math, os, pickle, subprocess, sys, threading, time, types
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')


def _limit_malloc_arenas(n=2):
    """glibc: cap per-thread malloc arenas (same effect as MALLOC_ARENA_MAX=2) so freed memory is reused."""
    try:
        import ctypes, ctypes.util
        ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6').mallopt(-8, n)   # M_ARENA_MAX
    except Exception:
        pass


_limit_malloc_arenas()
import numpy as np
from numba import njit, prange, set_num_threads, config as numba_config

VERSION = 'era5_iso_fast 1.1'
LEVELS = np.arange(15, 96)
PRESSURE = np.array([10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250]) * 100.
MODEL_URL = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1'
SURFACE_URL = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
FIELDS = {'T': 'temperature', 'U': 'u_component_of_wind', 'V': 'v_component_of_wind', 'OMEGA': 'vertical_velocity'}
EPOCH = np.datetime64('1900-01-01T00', 'h')
CHUNK_DECODED_BYTES = 18 * 721 * 1440 * 4

# Constants exactly as in the original core module.
R = 6_371_000.0
G = 9.80665
KAPPA = 287.058 / 1004.0
P0 = 100_000.0
DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

# ----------------------------------------------------------------------------------
# Verbatim from era5_monthly_diffusion_81.py (April13 tables, core module, accelerator)
# ----------------------------------------------------------------------------------
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

@dataclass
class Snapshot:
    time: np.datetime64
    lat: np.ndarray
    lon: np.ndarray
    p: np.ndarray  # (level,latitude,longitude), Pa, increasing vertically
    theta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray  # Pa/s


@dataclass
class Config:
    theta: np.ndarray
    pressure: np.ndarray
    p_top: float = 100.0
    p_bottom: float = 50_000.0
    max_latitude: float = 85.0
    min_zonal_coverage: float = 1.0
    min_time_coverage: float = 0.8
    spinup_hours: float = 0.0
    reset: str = "monthly"
    trajectory_iterations: int = 2
    chunk: int = 100_000
    min_path_coverage: float = 0.99


def mass_mean(field, sigma, minimum_coverage=1.0):
    available = np.isfinite(sigma) & (sigma > 0)
    valid = available & np.isfinite(field)
    sw = np.sum(np.where(valid,sigma,0.0),axis=-1,dtype=np.float64)
    reference = np.sum(np.where(available,sigma,0.0),axis=-1,dtype=np.float64)
    geographic = valid.mean(axis=-1)
    with np.errstate(divide="ignore",invalid="ignore"):
        mean = np.sum(np.where(valid,field*sigma,0.0),axis=-1,dtype=np.float64)/sw
        fraction = sw/reference
    okay = (geographic+1e-10 >= minimum_coverage) & (sw > 0)
    return np.where(okay,mean,np.nan), np.where(reference>0,fraction,0.0), geographic


def tensor(state, eta, xi, cfg, support=None):
    if support is None:support=np.ones_like(eta)
    joint = np.isfinite(eta+xi+state["vp"]+state["Qp"]) & (support>0)
    effective_sigma=state["sigma"]*np.where(joint,support,0.)
    full_mass=np.nansum(state["sigma"],axis=-1,dtype=np.float64)
    supported_mass=np.nansum(effective_sigma,axis=-1,dtype=np.float64)
    path_fraction=np.divide(supported_mass,full_mass,out=np.zeros_like(full_mass),where=full_mass>0)
    moments = {}
    for name, field in (("K_yy",state["vp"]*eta), ("K_ytheta",state["vp"]*xi),
                         ("K_thetay",state["Qp"]*eta), ("K_thetatheta",state["Qp"]*xi)):
        value, _, geo = mass_mean(np.where(joint,field,np.nan),effective_sigma,cfg.min_zonal_coverage)
        moments[name]=np.where(path_fraction+1e-10>=cfg.min_path_coverage,value,np.nan)
    moments.update(D_yy=moments["K_yy"],
                   D_ytheta=0.5*(moments["K_ytheta"]+moments["K_thetay"]),
                   D_thetatheta=moments["K_thetatheta"],
                   A_ytheta=0.5*(moments["K_ytheta"]-moments["K_thetay"]),
                   displacement_mass_coverage=path_fraction, displacement_zonal_coverage=geo,
                   displacement_interpolation_support=np.mean(np.where(joint,support,0.),axis=-1))
    return moments


def transform_tensor(Dyy, Dytheta, Dthetatheta, slope, r):
    """Exact second-order tensor transformation for the specified 2-D map."""
    cross_tilt = slope*Dyy
    cross_theta = r*Dytheta
    vertical_tilt = slope*slope*Dyy
    vertical_mixed = 2*slope*r*Dytheta
    vertical_theta = r*r*Dthetatheta
    return dict(D_yy=Dyy, D_yp=cross_tilt+cross_theta,
                D_pp=vertical_tilt+vertical_mixed+vertical_theta,
                D_yp_tilt=cross_tilt, D_yp_theta_cross=cross_theta,
                D_pp_tilt=vertical_tilt, D_pp_mixed=vertical_mixed,
                D_pp_theta=vertical_theta,
                D_pp_geometry_correction=vertical_tilt+vertical_mixed)


class Moments:
    """Time-weighted mean and temporal variability; not a standard error."""
    def __init__(self):
        self.fields = {}
        self.duration = 0.0
        self.samples = 0

    def add(self, fields, dt):
        self.duration += dt
        self.samples += 1
        for name,value in fields.items():
            value=np.asarray(value,dtype=np.float64)
            if name not in self.fields:
                self.fields[name]=[np.zeros_like(value),np.zeros_like(value),np.zeros_like(value),np.zeros_like(value)]
            weight, mean, m2, count=self.fields[name]
            valid=np.isfinite(value)
            new=weight+dt*valid
            delta=np.where(valid,value-mean,0)
            updated=mean+np.divide(dt*delta,new,out=np.zeros_like(value),where=new>0)
            m2 += np.where(valid,dt*delta*(value-updated),0)
            mean[:] = updated
            weight[:] = new
            count += valid

    def finish(self, minimum):
        out={}
        for name,(weight,mean,m2,count) in self.fields.items():
            coverage=weight/self.duration if self.duration else np.zeros_like(weight)
            valid=(coverage+1e-10>=minimum)&(weight>0)
            out[name]=np.where(valid,mean,np.nan)
            # Welford roundoff can put a nominally zero variance just below 0.
            variance=np.divide(m2,weight,out=np.zeros_like(m2),where=weight>0)
            out[name+"_std"]=np.where(valid,np.sqrt(np.maximum(variance,0.)),np.nan)
            out[name+"_time_coverage"]=coverage
            out[name+"_n"]=count.astype(np.int32)
        return out


@njit(parallel=True, fastmath=False, cache=True)
def remap_kernel(p, th, fields, targets, lat, ptop, pbottom, maxlat, gravity):
    nz, ny, nx = p.shape
    out = np.full((5,len(targets),ny,nx), np.nan, np.float32)
    for col in prange(ny*nx):
        j, i = col//nx, col % nx
        if abs(lat[j]) > maxlat:
            continue
        count = np.zeros(len(targets), np.int32)
        index = np.zeros(len(targets), np.int32)
        for k in range(nz-1):
            a, b = th[k,j,i], th[k+1,j,i]
            delta = b-a
            if not (p[k,j,i] <= pbottom and p[k+1,j,i] >= ptop and np.isfinite(delta)):
                continue
            if a > b:
                low = np.searchsorted(targets,b,side='right')
                high = np.searchsorted(targets,a,side='right')
            elif a < b:
                low = np.searchsorted(targets,a,side='left')
                high = np.searchsorted(targets,b,side='left')
            else:
                continue
            for t in range(low, high):
                count[t] += 1
                index[t] = k
        for t in range(len(targets)):
            if count[t] != 1:
                continue
            k = index[t]
            dtheta = th[k+1,j,i]-th[k,j,i]
            if dtheta >= 0:
                continue
            f = (targets[t]-th[k,j,i])/dtheta
            dp = p[k+1,j,i]-p[k,j,i]
            pressure = p[k,j,i]+f*dp
            sigma = -dp/dtheta/gravity
            if not (ptop <= pressure <= pbottom and np.isfinite(sigma) and sigma > 0):
                continue
            out[0,t,j,i] = pressure
            out[4,t,j,i] = sigma
            for a in range(3):
                v1, v2 = fields[a][k,j,i], fields[a][k+1,j,i]
                out[a+1,t,j,i] = v1+f*(v2-v1)
    return out


def remap(frame, cfg):
    """The accelerator's remap (unchanged)."""
    out = remap_kernel(frame.p, frame.theta, (frame.u, frame.v, frame.omega), cfg.theta, frame.lat,
                       cfg.p_top, cfg.p_bottom, cfg.max_latitude, 9.80665)
    return {n: out[k] for k, n in enumerate(('p', 'u', 'v', 'omega', 'sigma'))}


# ----------------------------------------------------------------------------------
# Configuration identical to run_locked(): cfg = Config(theta 300..1000/5, targets, ...)
# ----------------------------------------------------------------------------------
def model_reference_pressure_hpa():
    """April13 select_levels(1, 500): fixed A + B*101325 Pa full-level reference pressures."""
    a_full = 0.5 * (A_HALF[:-1] + A_HALF[1:])
    b_full = 0.5 * (B_HALF[:-1] + B_HALF[1:])
    p_full_hpa = (a_full + b_full * 101325.0) / 100.0
    sel = np.where((p_full_hpa >= 1.0) & (p_full_hpa <= 500.0))[0]
    if not np.array_equal(sel + 1, LEVELS):
        raise ValueError('Reference pressure selection is not ERA5 levels 15-95')
    return p_full_hpa[sel]


def make_config():
    pref = model_reference_pressure_hpa()
    cfg = Config(np.arange(300., 1000.01, 5.), np.unique(np.r_[PRESSURE, pref * 100]),
                 min_time_coverage=0, min_path_coverage=0, reset='never')
    return cfg, pref


# ----------------------------------------------------------------------------------
# Fused Numba kernels. Each reproduces CORE.departures (+ CORE.advance_displacements)
# with the accelerator's native/isentropic sampling, operation by operation and with the
# same float32/float64 types as the NumPy original (NEP 50 float32 chains for v and omega,
# float64 for u). Only the work layout changes.
# ----------------------------------------------------------------------------------
@njit(inline='always')
def _mod360(x):
    """Bitwise equal to Python/NumPy float x % 360. for every finite x (fmod semantics)."""
    if 0.0 <= x < 360.0:
        return x + 0.0
    if -360.0 <= x < 0.0:
        return x + 360.0
    if 360.0 <= x < 720.0:
        return x - 360.0
    return x % 360.


@njit(inline='always')
def _right(arr, x, g):
    """== np.searchsorted(arr, x, side='right') - 1 for sorted arr, starting from guess g."""
    n = arr.shape[0]
    j = min(max(g, -1), n - 1)
    while j + 1 < n and arr[j + 1] <= x:
        j += 1
    while j >= 0 and arr[j] > x:
        j -= 1
    return j


@njit(inline='always')
def _horizontal(lat, lon, la, lo, ilat, ilon):
    lo = _mod360(lo)
    ny = lat.shape[0]
    nx = lon.shape[0]
    j = _right(lat, la, int(math.floor((la - lat[0]) * ilat)))
    j = min(max(j, 0), ny - 2)
    i = _right(lon, lo, int(math.floor((lo - lon[0]) * ilon))) % nx
    ip = (i + 1) % nx
    fy = (la - lat[j]) / (lat[j + 1] - lat[j])
    fx = _mod360(lo - lon[i]) / _mod360(lon[ip] - lon[i])
    return j, i, ip, fy, fx


@njit(inline='always')
def _bracket(p, jj, ii, target, g):
    """Same 'lower' as the original binary search on a strictly increasing column."""
    nlev = p.shape[0]
    k = min(max(g, 0), nlev - 2)
    if p[k, jj, ii] <= target:
        while k < nlev - 2 and p[k + 1, jj, ii] <= target:
            k += 1
    else:
        while k > 0 and p[k, jj, ii] > target:
            k -= 1
    return k


@njit(inline='always')
def _native3(p, fa, fb, fc, lat, lon, ilat, ilon, la, lo, target, g):
    nanf = np.float32(np.nan)
    if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(target) and lat[0] <= la <= lat[-1]):
        return nanf, nanf, nanf, g
    j, i, ip, fy, fx = _horizontal(lat, lon, la, lo, ilat, ilon)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    ok0 = True
    ok1 = True
    ok2 = True
    for c in range(4):
        jj = j + c // 2
        ii = i if c % 2 == 0 else ip
        w = (1 - fy if c < 2 else fy) * (1 - fx if c % 2 == 0 else fx)
        if w <= 1e-12:
            continue
        lower = _bracket(p, jj, ii, target, g)
        g = lower
        p1 = p[lower, jj, ii]
        p2 = p[lower + 1, jj, ii]
        supported = p[0, jj, ii] <= target <= p[-1, jj, ii] and np.isfinite(p1) and np.isfinite(p2) and p2 > p1
        if not supported:
            ok0 = False
            ok1 = False
            ok2 = False
            continue
        f = (target - p1) / (p2 - p1)
        v1 = fa[lower, jj, ii]
        v2 = fa[lower + 1, jj, ii]
        value = v1 + f * (v2 - v1)
        if np.isfinite(value):
            s0 += w * value
        else:
            ok0 = False
        v1 = fb[lower, jj, ii]
        v2 = fb[lower + 1, jj, ii]
        value = v1 + f * (v2 - v1)
        if np.isfinite(value):
            s1 += w * value
        else:
            ok1 = False
        v1 = fc[lower, jj, ii]
        v2 = fc[lower + 1, jj, ii]
        value = v1 + f * (v2 - v1)
        if np.isfinite(value):
            s2 += w * value
        else:
            ok2 = False
    return (np.float32(s0) if ok0 else nanf), (np.float32(s1) if ok1 else nanf), (np.float32(s2) if ok2 else nanf), g


@njit(inline='always')
def _native1(p, fa, lat, lon, ilat, ilon, la, lo, target, g):
    nanf = np.float32(np.nan)
    if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(target) and lat[0] <= la <= lat[-1]):
        return nanf, g
    j, i, ip, fy, fx = _horizontal(lat, lon, la, lo, ilat, ilon)
    s0 = 0.0
    ok0 = True
    for c in range(4):
        jj = j + c // 2
        ii = i if c % 2 == 0 else ip
        w = (1 - fy if c < 2 else fy) * (1 - fx if c % 2 == 0 else fx)
        if w <= 1e-12:
            continue
        lower = _bracket(p, jj, ii, target, g)
        g = lower
        p1 = p[lower, jj, ii]
        p2 = p[lower + 1, jj, ii]
        supported = p[0, jj, ii] <= target <= p[-1, jj, ii] and np.isfinite(p1) and np.isfinite(p2) and p2 > p1
        if not supported:
            ok0 = False
            continue
        f = (target - p1) / (p2 - p1)
        v1 = fa[lower, jj, ii]
        v2 = fa[lower + 1, jj, ii]
        value = v1 + f * (v2 - v1)
        if np.isfinite(value):
            s0 += w * value
        else:
            ok0 = False
    return (np.float32(s0) if ok0 else nanf), g


@njit(inline='always')
def _departure(sp, su, sv, so, latj, loni, cj, pp, pu, pv, pw, pth, lat, lon, ilat, ilon,
               dt32, R32, DEG32, c64, c32, h32, maxlat, g):
    """CORE.departures for one arrival point -> (lat_dep, lon_dep_unwrapped, theta_dep f32, g)."""
    la = latj - np.float64(((sv * dt32) / R32) * DEG32)
    lo = loni - (np.float64(su * dt32) / (R * cj)) * DEG
    pr = sp - so * dt32
    for _ in range(2):
        wu, wv, wo, g = _native3(pp, pu, pv, pw, lat, lon, ilat, ilon, la, lo, np.float64(pr), g)
        old_cos = np.cos(la * DEG2RAD)
        lo = loni - (c64 * (np.float64(su) / cj + np.float64(wu) / old_cos)) * DEG
        la = latj - np.float64((c32 * (sv + wv)) * DEG32)
        pr = sp - h32 * (so + wo)
    nan = np.nan
    if abs(la) <= maxlat and np.isfinite(la + lo + np.float64(pr)):
        th, g = _native1(pp, pth, lat, lon, ilat, ilon, la, lo, np.float64(pr), g)
        return la, lo, th, g
    return nan, nan, np.float32(nan), g


@njit(parallel=True, cache=True)
def hourly_q_kernel(s_p, s_u, s_v, s_o, theta_t, lat, lon, coslat, pp, pu, pv, pw, pth,
                    dt, dt32, R32, DEG32, c64, c32, h32, maxlat, out):
    nt, ny, nx = s_p.shape
    ilat = 1.0 / (lat[1] - lat[0])
    ilon = 1.0 / (lon[1] - lon[0])
    nlev = pp.shape[0]
    for row in prange(nt * ny):
        k = row // ny
        j = row - k * ny
        latj = lat[j]
        cj = coslat[j]
        thk = theta_t[k]
        g = nlev // 2
        for i in range(nx):
            la, lo, th, g = _departure(s_p[k, j, i], s_u[k, j, i], s_v[k, j, i], s_o[k, j, i], latj, lon[i], cj,
                                       pp, pu, pv, pw, pth, lat, lon, ilat, ilon,
                                       dt32, R32, DEG32, c64, c32, h32, maxlat, g)
            out[k, j, i] = (thk - np.float64(th)) / dt


@njit(parallel=True, cache=True)
def halfstep_kernel(s_p, s_u, s_v, s_o, theta_t, lat, lon, coslat, pp, pu, pv, pw, pth,
                    dt, dt32, R32, DEG32, c64, c32, h32, maxlat,
                    sup, eta, xi, vp_old, qp_old, vp_new, qp_new, out_eta, out_xi, out_w):
    """CORE.departures + CORE.advance_displacements, fused per arrival point."""
    nt, ny, nx = s_p.shape
    ilat = 1.0 / (lat[1] - lat[0])
    ilon = 1.0 / (lon[1] - lon[0])
    ith = 1.0 / (theta_t[1] - theta_t[0])
    nlev = pp.shape[0]
    half = 0.5 * dt
    nanf = np.float32(np.nan)
    zero32 = np.float32(0.0)
    one32 = np.float32(1.0)
    tiny32 = np.float32(1e-12)
    for row in prange(nt * ny):
        k = row // ny
        j = row - k * ny
        latj = lat[j]
        cj = coslat[j]
        g = nlev // 2
        for i in range(nx):
            la, lo, th, g = _departure(s_p[k, j, i], s_u[k, j, i], s_v[k, j, i], s_o[k, j, i], latj, lon[i], cj,
                                       pp, pu, pv, pw, pth, lat, lon, ilat, ilon,
                                       dt32, R32, DEG32, c64, c32, h32, maxlat, g)
            dlo = _mod360(lo)
            qt = np.float64(th)
            o_s = nanf
            o_e = nanf
            o_x = nanf
            o_v = nanf
            o_q = nanf
            if np.isfinite(la) and np.isfinite(dlo) and np.isfinite(qt) and lat[0] <= la <= lat[-1] and theta_t[0] <= qt <= theta_t[-1]:
                j2, i2, ip2, fy, fx = _horizontal(lat, lon, la, dlo, ilat, ilon)
                k2 = _right(theta_t, qt, int(math.floor((qt - theta_t[0]) * ith)))
                k2 = min(max(k2, 0), nt - 2)
                fz = (qt - theta_t[k2]) / (theta_t[k2 + 1] - theta_t[k2])
                ss = 0.0
                se = 0.0
                sx = 0.0
                sv_ = 0.0
                sq = 0.0
                ks = True
                ke = True
                kx = True
                kv = True
                kq = True
                for c in range(4):
                    jj = j2 + c // 2
                    ii = i2 if c % 2 == 0 else ip2
                    wxy = (1 - fy if c < 2 else fy) * (1 - fx if c % 2 == 0 else fx)
                    for z in range(2):
                        w = wxy * (1 - fz if z == 0 else fz)
                        if w <= 1e-12:
                            continue
                        kk = k2 + z
                        s = sup[kk, jj, ii]
                        v_s = np.float64(s)
                        if s > 0:
                            v_e = np.float64(s * eta[kk, jj, ii])
                            v_x = np.float64(s * xi[kk, jj, ii])
                            v_v = np.float64(s) * vp_old[kk, jj, ii]
                            v_q = np.float64(s) * qp_old[kk, jj, ii]
                        else:
                            v_e = 0.0
                            v_x = 0.0
                            v_v = 0.0
                            v_q = 0.0
                        if np.isfinite(v_s):
                            ss += w * v_s
                        else:
                            ks = False
                        if np.isfinite(v_e):
                            se += w * v_e
                        else:
                            ke = False
                        if np.isfinite(v_x):
                            sx += w * v_x
                        else:
                            kx = False
                        if np.isfinite(v_v):
                            sv_ += w * v_v
                        else:
                            kv = False
                        if np.isfinite(v_q):
                            sq += w * v_q
                        else:
                            kq = False
                if ks:
                    o_s = np.float32(ss)
                if ke:
                    o_e = np.float32(se)
                if kx:
                    o_x = np.float32(sx)
                if kv:
                    o_v = np.float32(sv_)
                if kq:
                    o_q = np.float32(sq)
            vpn = vp_new[k, j, i]
            qpn = qp_new[k, j, i]
            wgt = zero32
            if np.isfinite(vpn + qpn + np.float64(o_s)):
                wgt = min(max(o_s, zero32), one32)
            en = np.float64(o_e) + half * (np.float64(o_v) + np.float64(wgt) * vpn)
            xn = np.float64(o_x) + half * (np.float64(o_q) + np.float64(wgt) * qpn)
            if wgt > tiny32 and np.isfinite(en + xn):
                out_w[k, j, i] = wgt
                out_eta[k, j, i] = np.float32(en / np.float64(wgt))
                out_xi[k, j, i] = np.float32(xn / np.float64(wgt))
            else:
                out_w[k, j, i] = zero32
                out_eta[k, j, i] = zero32
                out_xi[k, j, i] = zero32


def _consts(dt):
    c64 = 0.5 * dt / R
    return dict(dt=float(dt), dt32=np.float32(dt), R32=np.float32(R), DEG32=np.float32(DEG),
                c64=c64, c32=np.float32(c64), h32=np.float32(0.5 * dt))


def hourly_Q(prev, state, cfg, dt=3600.):
    """Q = (theta_target - theta at the departure point)/dt, as CORE.departures(...)['Q']."""
    out = np.empty(state['p'].shape, np.float64)
    hourly_q_kernel(state['p'], state['u'], state['v'], state['omega'], cfg.theta, prev.lat, prev.lon,
                    np.cos(np.deg2rad(prev.lat)), prev.p, prev.u, prev.v, prev.omega, prev.theta,
                    maxlat=float(cfg.max_latitude), out=out, **_consts(dt))
    return out


def halfstep(oldframe, oldstate, newstate, iso_state, cfg, dt=1800.):
    """One 30-minute step: CORE.departures + CORE.advance_displacements -> (eta, xi, support)."""
    eta, xi, support = iso_state
    shape = newstate['p'].shape
    out = [np.empty(shape, np.float32) for _ in range(3)]
    halfstep_kernel(newstate['p'], newstate['u'], newstate['v'], newstate['omega'], cfg.theta, oldframe.lat, oldframe.lon,
                    np.cos(np.deg2rad(oldframe.lat)), oldframe.p, oldframe.u, oldframe.v, oldframe.omega, oldframe.theta,
                    maxlat=float(cfg.max_latitude), sup=support, eta=eta, xi=xi,
                    vp_old=oldstate['vp'], qp_old=oldstate['Qp'], vp_new=newstate['vp'], qp_new=newstate['Qp'],
                    out_eta=out[0], out_xi=out[1], out_w=out[2], **_consts(dt))
    return tuple(out)


# ----------------------------------------------------------------------------------
# NumPy steps: theta-slab threads (every reduction is along longitude within one row)
# ----------------------------------------------------------------------------------
class Slabs:
    def __init__(self, threads):
        self.threads = max(1, int(threads))
        self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix='slab') if self.threads > 1 else None

    def map(self, fn, rows):
        if self.pool is None:
            return [fn(slice(0, rows))]
        k = min(rows, 2 * self.threads)
        edges = np.linspace(0, rows, k + 1).astype(int)
        return list(self.pool.map(fn, [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]))


SERIAL = Slabs(1)


def prepare_means(state, Q, cfg, slabs):
    """CORE.prepare_means evaluated per theta slab."""
    shape = Q.shape
    v_mean = np.empty(shape[:2])
    Q_mean = np.empty(shape[:2])
    vp = np.empty(shape, np.float64)
    Qp = np.empty(shape, np.float64)

    def work(sl):
        mv, _, _ = mass_mean(state['v'][sl], state['sigma'][sl], cfg.min_zonal_coverage)
        mq, _, _ = mass_mean(Q[sl], state['sigma'][sl], cfg.min_zonal_coverage)
        v_mean[sl] = mv
        Q_mean[sl] = mq
        vp[sl] = state['v'][sl] - mv[..., None]
        Qp[sl] = Q[sl] - mq[..., None]
    slabs.map(work, shape[0])
    state.update(Q=Q, v_mean=v_mean, Q_mean=Q_mean, vp=vp, Qp=Qp)


def tensor_slabs(state, iso_state, cfg, slabs):
    """CORE.tensor(state, eta, xi, cfg, support) evaluated per theta slab."""
    eta, xi, support = iso_state

    def work(sl):
        return tensor({'vp': state['vp'][sl], 'Qp': state['Qp'][sl], 'sigma': state['sigma'][sl]},
                      eta[sl], xi[sl], cfg, support[sl])
    parts = slabs.map(work, eta.shape[0])
    return {k: (parts[0][k] if len(parts) == 1 else np.concatenate([q[k] for q in parts], axis=0)) for k in parts[0]}


def _brackets(P, valid, pressure):
    """Last theta bracket containing each target pressure (same overwrite order as the loop)."""
    ok = valid[:-1] & valid[1:] & (P[:-1] > P[1:])
    kk = np.full((len(pressure), P.shape[1]), -1, np.intp)
    pr = pressure[:, None]
    for k in range(P.shape[0] - 1):
        kk[ok[k] & (pr <= P[k]) & (pr >= P[k + 1])] = k
    return kk


def _interp(values, P, kk, pressure):
    hit = kk >= 0
    k = np.where(hit, kk, 0)
    j = np.broadcast_to(np.arange(P.shape[1]), kk.shape)
    with np.errstate(all='ignore'):
        p0 = P[k, j]
        p1 = P[k + 1, j]
        f = (pressure[:, None] - p0) / (p1 - p0)
        v0 = values[k, j]
        v1 = values[k + 1, j]
        out = v0 + f * (v1 - v0)
    return np.where(hit, out, np.nan)


def map_to_pressure(state, moments, lat, cfg):
    """CORE.map_to_pressure with the (latitude, theta) Python loops vectorized; bitwise identical."""
    good = np.isfinite(state["p"])
    count = good.sum(axis=-1)
    pbar = np.divide(np.where(good, state["p"], 0).sum(axis=-1, dtype=np.float64),
                     count, out=np.full(count.shape, np.nan), where=count > 0)
    pbar[count / state["p"].shape[-1] + 1e-10 < cfg.min_zonal_coverage] = np.nan
    r = np.gradient(pbar, cfg.theta, axis=0, edge_order=1)
    slope = np.gradient(pbar, R * np.deg2rad(lat), axis=1, edge_order=1)
    valid = (r < 0) & np.isfinite(r + slope + moments["D_yy"] + moments["D_ytheta"] + moments["D_thetatheta"])
    mapped = transform_tensor(moments["D_yy"], moments["D_ytheta"], moments["D_thetatheta"], slope, r)
    mapped["A_yp"] = r * moments["A_ytheta"]
    mapped["K_yp"] = mapped["D_yp"] + mapped["A_yp"]
    mapped["K_py"] = mapped["D_yp"] - mapped["A_yp"]
    mapped["slope"] = slope
    mapped["r"] = r
    kk = _brackets(pbar, valid, cfg.pressure)
    pressure_fields = {n: _interp(v, pbar, kk, cfg.pressure) for n, v in mapped.items()}
    moments.update(p_mean=pbar, dp_dtheta=r, dp_dy_theta=slope,
                   v_mass_mean=state["v_mean"], Q_mass_mean=state["Q_mean"],
                   sigma_mean=np.mean(state["sigma"], axis=-1))
    return pressure_fields


def mean_first(native_means, lat, cfg):
    """TERMS.mapped_from_means without the H/L/S pieces: map the mean tensor with the mean isentrope."""
    m = native_means
    native = {k: m[k].copy() for k in ['D_yy', 'D_ytheta', 'D_thetatheta', 'A_ytheta']}
    state = {'p': m['p_mean'][..., None], 'sigma': m['sigma_mean'][..., None], 'v_mean': m['v_mass_mean'], 'Q_mean': m['Q_mass_mean']}
    return map_to_pressure(state, native, lat, cfg), native


# ----------------------------------------------------------------------------------
# Snapshot construction: CORE.snapshot_from_dataset arithmetic for ARCO input
# ----------------------------------------------------------------------------------
def build_frame(t, lat, lon, T, U, V, W, PS, aa, bb):
    T = np.array(T, dtype=np.float32)
    u = np.array(U, dtype=np.float32)
    v = np.array(V, dtype=np.float32)
    w = np.array(W, dtype=np.float32)
    if not np.all(np.isfinite(T)) or np.nanmin(T) < 80 or np.nanmax(T) > 400:
        raise ValueError(f'Temperature must be finite and in kelvin (80..400 K): {t}')
    ps = np.asarray(PS).astype(float)
    if np.nanmedian(ps) < 20_000 or np.nanmedian(ps) > 120_000:
        raise ValueError(f'Implausible surface pressure: {t}')
    a = np.asarray(aa).astype(float)
    b = np.asarray(bb).astype(float)
    p = a[:, None, None] + b[:, None, None] * ps[None, :, :]
    if np.nanmedian(p[0]) > np.nanmedian(p[-1]):
        raise ValueError('Unexpected vertical orientation of the model levels')
    if not np.all(np.isfinite(p)) or np.any(p <= 0) or np.any(np.diff(p, axis=0) <= 0):
        raise ValueError('Physical pressure must be finite, positive and increasing in every column')
    theta = (T * (P0 / p) ** KAPPA).astype(np.float32)
    return Snapshot(np.datetime64(t, 'ns'), lat, lon, p, theta, u, v, w)


def midpoint(prev, now):
    return Snapshot(prev.time + np.timedelta64(30, 'm'), prev.lat, prev.lon,
                    *[(getattr(prev, n) + getattr(now, n)) * .5 for n in ['p', 'theta', 'u', 'v', 'omega']])


# ----------------------------------------------------------------------------------
# ERA5 reader: direct Zarr chunk reads, parallel, prefetched, subsampled on decode
# ----------------------------------------------------------------------------------
def fetch_bytes(url, verify_md5=False, timeout=180, retries=8):
    if not url.startswith(('http://', 'https://')):
        with open(url[7:] if url.startswith('file://') else url, 'rb') as stream:
            return stream.read()
    error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
            if 'content-length' in headers and len(data) != int(headers['content-length']):
                raise IOError(f'truncated read: {len(data)} of {headers["content-length"]} bytes')
            if verify_md5:
                for item in headers.get('x-goog-hash', '').split(','):
                    item = item.strip()
                    if item.startswith('md5=') and base64.b64encode(hashlib.md5(data).digest()).decode() != item[4:]:
                        raise IOError('md5 mismatch')
            return data
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                raise
            error = exc
        except Exception as exc:
            error = exc
        time.sleep(min(2 ** attempt, 30))
    raise IOError(f'Could not read {url} after {retries} attempts: {error}')


class ZarrStore:
    """Read-only consolidated Zarr v2 store over HTTPS (or a local copy, for testing)."""
    def __init__(self, root, verify_md5=False):
        self.root = root.rstrip('/')
        self.verify_md5 = verify_md5
        self.meta = json.loads(fetch_bytes(self.root + '/.zmetadata'))['metadata']
        self._time = {}
        self._lock = threading.Lock()

    def info(self, name):
        return self.meta[name + '/.zarray']

    def attrs(self, name):
        return self.meta.get(name + '/.zattrs', {})

    def chunk(self, name, index):
        from numcodecs import get_codec
        info = self.info(name)
        key = info.get('dimension_separator', '.').join(map(str, index))
        raw = fetch_bytes(f'{self.root}/{name}/{key}', self.verify_md5)
        data = get_codec(info['compressor']).decode(raw) if info.get('compressor') else raw
        for spec in reversed(info.get('filters') or []):
            data = get_codec(spec).decode(data)
        array = np.frombuffer(data, dtype=np.dtype(info['dtype']))
        if array.size != int(np.prod(info['chunks'])):
            raise ValueError(f'Unexpected decoded size for {name}/{key}')
        return array.reshape(info['chunks'], order=info.get('order', 'C')), len(raw)

    def coordinate(self, name):
        info = self.info(name)
        n, width = info['shape'][0], info['chunks'][0]
        return np.concatenate([self.chunk(name, (i,))[0] for i in range(-(-n // width))])[:n]

    def time_value(self, hour):
        width = self.info('time')['chunks'][0]
        with self._lock:
            if hour // width not in self._time:
                self._time[hour // width] = self.chunk('time', (hour // width,))[0]
            return int(self._time[hour // width][hour % width])


class Reader:
    """Hourly snapshots of T, U, V, omega (levels 15-95) and PS on the 0.5 x 1 degree grid."""
    def __init__(self, times, a):
        self.times = times
        self.batch_hours = a.batch_hours
        self.prefetch = a.prefetch_batches
        self.stride = a.test_lon_stride
        self.model = ZarrStore(a.model_url, a.verify_md5)
        self.surface = ZarrStore(a.surface_url, a.verify_md5)
        m, s = self.model, self.surface
        for name in FIELDS.values():
            if m.info(name)['chunks'] != [1, 18, 721, 1440]:
                raise ValueError(f'Unexpected chunking for {name}: {m.info(name)["chunks"]}')
        if s.info('surface_pressure')['chunks'] != [1, 721, 1440]:
            raise ValueError('Unexpected chunking for surface_pressure')
        pv = np.asarray(m.attrs('temperature')['GRIB_pv'], dtype=float)
        if pv.shape != (276,):
            raise ValueError('Expected 137-level ERA5 hybrid coefficients')
        self.aa = (.5 * (pv[:137] + pv[1:138]))[LEVELS - 1]
        self.bb = (.5 * (pv[138:-1] + pv[139:]))[LEVELS - 1]
        if s.attrs('surface_pressure').get('units') != 'Pa':
            raise ValueError('ARCO surface pressure must be Pa')
        if m.attrs('vertical_velocity').get('units') not in ['Pa s**-1', 'Pa s-1', 'Pa/s']:
            raise ValueError('Unexpected ARCO omega units')
        lat, lon = m.coordinate('latitude'), m.coordinate('longitude')
        np.testing.assert_array_equal(lat, np.arange(90, -90.01, -.25))
        np.testing.assert_array_equal(lon, np.arange(0, 360, .25))
        np.testing.assert_array_equal(m.coordinate('hybrid'), np.arange(1, 138))
        np.testing.assert_array_equal(s.coordinate('latitude'), lat)
        np.testing.assert_array_equal(s.coordinate('longitude'), lon)
        self.lat = lat[::2][::-1].astype(float)
        self.lon = lon[::self.stride].astype(float)
        np.testing.assert_array_equal(self.lat, np.arange(-90, 90.01, .5))
        np.testing.assert_array_equal(self.lon, np.arange(0, 360, .25 * self.stride))
        self.pool = ThreadPoolExecutor(a.download_workers, thread_name_prefix='era5')
        self.assembler = ThreadPoolExecutor(1, thread_name_prefix='batch')
        self.futures = {}
        self.wait_seconds = 0.
        self.stats = {'chunks': 0, 'bytes': 0, 'batch_seconds': 0., 'batches': 0}
        self._lock = threading.Lock()

    def _level_chunk(self, name, hour, k, dest, j):
        data, nbytes = self.model.chunk(name, (hour, k, 0, 0))
        lo, hi = max(14, k * 18), min(95, (k + 1) * 18)
        dest[j, lo - 14:hi - 14] = data[0, lo - k * 18:hi - k * 18, ::2, ::self.stride][:, ::-1, :]
        return nbytes

    def _ps_chunk(self, hour, dest, j):
        data, nbytes = self.surface.chunk('surface_pressure', (hour, 0, 0))
        dest[j] = data[0, ::2, ::self.stride][::-1, :]
        return nbytes

    def _assemble(self, b):
        started = time.perf_counter()
        first = b * self.batch_hours
        last = min(first + self.batch_hours, len(self.times))
        shape = (last - first, len(LEVELS), len(self.lat), len(self.lon))
        fields = {s: np.empty(shape, np.float32) for s in FIELDS}
        ps = np.empty((shape[0],) + shape[2:], np.float32)
        jobs = []
        for j in range(last - first):
            hour = int((np.datetime64(self.times[first + j], 'h') - EPOCH) / np.timedelta64(1, 'h'))
            for store in (self.model, self.surface):
                if store.time_value(hour) != hour:
                    raise ValueError(f'Time index mismatch in {store.root} for {self.times[first + j]}')
            for short, name in FIELDS.items():
                for k in range(6):
                    jobs.append(self.pool.submit(self._level_chunk, name, hour, k, fields[short], j))
            jobs.append(self.pool.submit(self._ps_chunk, hour, ps, j))
        nbytes = sum(job.result() for job in jobs)
        with self._lock:
            self.stats['chunks'] += len(jobs)
            self.stats['bytes'] += nbytes
            self.stats['batch_seconds'] += time.perf_counter() - started
            self.stats['batches'] += 1
        return fields, ps

    def raw(self, t):
        t = np.datetime64(t, 'h')
        i = int(np.searchsorted(self.times, t))
        if i >= len(self.times) or self.times[i] != t:
            raise ValueError(f'{t} is outside the selected input interval')
        b, j = divmod(i, self.batch_hours)
        nbatches = -(-len(self.times) // self.batch_hours)
        for c in range(b, min(nbatches, b + 1 + self.prefetch)):
            if c not in self.futures:
                self.futures[c] = self.assembler.submit(self._assemble, c)
        for c in [c for c in self.futures if c < b]:
            del self.futures[c]
        started = time.perf_counter()
        fields, ps = self.futures[b].result()
        self.wait_seconds += time.perf_counter() - started
        return fields['T'][j], fields['U'][j], fields['V'][j], fields['OMEGA'][j], ps[j]

    def frame(self, t):
        return build_frame(np.datetime64(t, 'h'), self.lat, self.lon, *self.raw(t), self.aa, self.bb)

    def close(self):
        self.futures.clear()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.assembler.shutdown(wait=False, cancel_futures=True)


class SyntheticReader:
    """Stably stratified, zonally wavy fields on levels 15-95 (no network; for --mode validate --synthetic)."""
    def __init__(self, times, a):
        self.times = times
        self.lat = np.arange(-90, 90.01, .5)
        self.lon = np.arange(0, 360, .25 * a.test_lon_stride)
        self.aa = (0.5 * (A_HALF[:-1] + A_HALF[1:]))[LEVELS - 1]
        self.bb = (0.5 * (B_HALF[:-1] + B_HALF[1:]))[LEVELS - 1]
        self.wait_seconds = 0.
        self.stats = {'chunks': 0, 'bytes': 0, 'batch_seconds': 0., 'batches': 0}

    def raw(self, t):
        h = float((np.datetime64(t, 'h') - np.datetime64('2010-01-01T00', 'h')) / np.timedelta64(1, 'h'))
        ph = 2 * np.pi * h / 120.
        phi = np.deg2rad(self.lat)[None, :, None]
        lam = np.deg2rad(self.lon)[None, None, :]
        ps = 101325. - 1500 * np.cos(2 * phi[0]) + 300 * np.cos(3 * lam[0] - ph) * np.cos(phi[0])
        p = self.aa[:, None, None] + self.bb[:, None, None] * ps[None]
        z = 7.0 * np.log(101325. / p)
        Tstd = np.where(z < 11, 288.15 - 6.5 * z, np.where(z < 20, 216.65, np.where(z < 32, 216.65 + (z - 20), 228.65 + 2.8 * (z - 32))))
        wave = np.cos(3 * lam - ph) * np.cos(phi)
        ones = np.ones_like(p)
        return ((Tstd + 5 * np.cos(phi) ** 2 + 3 * wave).astype(np.float32),
                (10 + 25 * np.sin(2 * phi) ** 2 * ones + 5 * np.sin(3 * lam - ph) * np.cos(phi)).astype(np.float32),
                (5 * np.sin(3 * lam - ph + .3) * np.cos(phi) * ones).astype(np.float32),
                (.02 * (p / 50000.) * np.cos(3 * lam - ph + 1.) * np.cos(phi)).astype(np.float32),
                ps.astype(np.float32))

    def frame(self, t):
        return build_frame(np.datetime64(t, 'h'), self.lat, self.lon, *self.raw(t), self.aa, self.bb)

    def close(self):
        pass


# ----------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def savejson(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def peak_memory_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    except Exception:
        return float('nan')


def same(a, b):
    """Bitwise equality, NaN positions included."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind != 'f':
        return np.array_equal(a, b)
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a[~na].view(np.uint8), b[~nb].view(np.uint8)))


def result_name(start, end):
    """diffusion_iso_YYYY_MM.nc for a calendar month, else diffusion_iso_YYYYMMDD_HH_YYYYMMDD_HH.nc."""
    s, e = np.datetime64(start, 'h'), np.datetime64(end, 'h')
    month = s.astype('datetime64[M]')
    if s == month.astype('datetime64[h]') and e == (month + 1).astype('datetime64[h]'):
        return 'diffusion_iso_' + str(month).replace('-', '_') + '.nc'
    stamp = lambda t: str(t).replace('-', '').replace('T', '_')
    return f'diffusion_iso_{stamp(s)}_{stamp(e)}.nc'


def month_bounds(year, month):
    start = np.datetime64(f'{year:04d}-{month:02d}', 'M')
    return start.astype('datetime64[h]'), (start + 1).astype('datetime64[h]')


def required_times(a):
    start, end = np.datetime64(a.start, 'h'), np.datetime64(a.end, 'h')
    return np.arange(start - np.timedelta64(1, 'h'), end + np.timedelta64(1, 'h'), np.timedelta64(1, 'h'))


def settings(a):
    start, end = np.datetime64(a.start, 'ns'), np.datetime64(a.end, 'ns')
    if start != start.astype('datetime64[h]') or end != end.astype('datetime64[h]'):
        raise ValueError('Start/end must be whole UTC hours')
    hours = int((end - start) / np.timedelta64(1, 'h'))
    if hours < 1:
        raise ValueError('End must be after start')
    if min(a.threads, a.slab_threads, a.download_workers, a.batch_hours, a.checkpoint_every) < 1 or a.prefetch_batches < 0:
        raise ValueError('Invalid thread/worker/batch/checkpoint setting')
    return {'version': VERSION, 'start': str(start), 'end': str(end), 'hours': hours, 'method': 'isentropic',
            'step_minutes': 30, 'input_cadence_hours': 1, 'levels': LEVELS.tolist(), 'latitude_step': .5,
            'longitude_step': .25 * a.test_lon_stride, 'theta_start': 300., 'theta_end': 1000., 'theta_step': 5.,
            'trajectory_iterations': 2, 'min_zonal_coverage': 1., 'min_path_coverage': 0., 'min_time_coverage': 0.,
            'p_top_Pa': 100., 'p_bottom_Pa': 50_000., 'max_latitude': 85., 'reset': 'never',
            'model_url': a.model_url, 'surface_url': a.surface_url}


NATIVE_UNITS = {'K_yy': 'm2 s-1', 'K_ytheta': 'm K s-1', 'K_thetay': 'm K s-1', 'K_thetatheta': 'K2 s-1',
                'D_yy': 'm2 s-1', 'D_ytheta': 'm K s-1', 'D_thetatheta': 'K2 s-1', 'A_ytheta': 'm K s-1',
                'p_mean': 'Pa', 'dp_dtheta': 'Pa K-1', 'dp_dy_theta': 'Pa m-1', 'v_mass_mean': 'm s-1',
                'Q_mass_mean': 'K s-1', 'sigma_mean': 'kg m-2 K-1'}
PRESSURE_UNITS = {'D_yy': 'm2 s-1', 'K_yy': 'm2 s-1', 'D_yp': 'm Pa s-1', 'D_py': 'm Pa s-1', 'K_yp': 'm Pa s-1',
                  'K_py': 'm Pa s-1', 'A_yp': 'm Pa s-1', 'D_yp_tilt': 'm Pa s-1', 'D_yp_theta_cross': 'm Pa s-1',
                  'D_pp': 'Pa2 s-1', 'K_pp': 'Pa2 s-1', 'D_pp_tilt': 'Pa2 s-1', 'D_pp_mixed': 'Pa2 s-1',
                  'D_pp_theta': 'Pa2 s-1', 'D_pp_geometry_correction': 'Pa2 s-1', 'slope': 'Pa m-1', 'r': 'Pa K-1'}
LONG_NAMES = {'D_yp_tilt': 's*D_yy', 'D_yp_theta_cross': 'r*D_ytheta', 'D_pp_tilt': 's^2*D_yy',
              'D_pp_mixed': '2*s*r*D_ytheta', 'D_pp_theta': 'r^2*D_thetatheta',
              'D_pp_geometry_correction': 's^2*D_yy + 2*s*r*D_ytheta (not a difference from any existing D_pp)',
              'slope': 's = dp/dy on the isentrope', 'r': 'r = dp/dtheta', 'K_yy': 'alias of D_yy',
              'K_pp': 'alias of D_pp', 'D_py': 'alias of D_yp'}


def _means(moments):
    finished = moments.finish(0)
    return {k: v for k, v in finished.items() if not k.endswith(('_std', '_time_coverage', '_n'))}


def write_output(path, mom_native, mom_tf, lat, cfg, pref, attrs):
    import xarray as xr
    index_p = np.searchsorted(cfg.pressure, PRESSURE)
    index_m = np.searchsorted(cfg.pressure, pref * 100)
    if not (np.array_equal(cfg.pressure[index_p], PRESSURE) and np.array_equal(cfg.pressure[index_m], pref * 100)):
        raise ValueError('Output pressures are not on the target grid')
    ds = xr.Dataset(coords={'latitude': ('latitude', lat, {'units': 'degrees_north'}),
                            'theta': ('theta', cfg.theta, {'units': 'K'}),
                            'pressure': ('pressure', PRESSURE, {'units': 'Pa'}),
                            'model_pressure': ('model_pressure', pref * 100, {'units': 'Pa'}),
                            'model_level': ('model_pressure', LEVELS)})
    native = _means(mom_native)
    for name in ['K_yy', 'K_ytheta', 'K_thetay', 'K_thetatheta', 'D_yy', 'D_ytheta', 'D_thetatheta', 'A_ytheta']:
        ds['native_' + name] = (('theta', 'latitude'), native[name], {'units': NATIVE_UNITS[name]})
    mapped, geometry = mean_first(native, lat, cfg)
    groups = [('iso_mean_first', mapped), ('iso_transform_first', _means(mom_tf))]
    for prefix, fields in groups:
        fields = dict(fields)
        fields.update(D_py=fields['D_yp'], K_pp=fields['D_pp'], K_yy=fields['D_yy'])
        for name, value in fields.items():
            meta = {'units': PRESSURE_UNITS[name]}
            if name in LONG_NAMES:
                meta['long_name'] = LONG_NAMES[name]
            ds[prefix + '_' + name] = (('pressure', 'latitude'), value[index_p], meta)
            ds[prefix + '_on_model_' + name] = (('model_pressure', 'latitude'), value[index_m], meta)
    for name in ['p_mean', 'dp_dtheta', 'dp_dy_theta']:
        ds['mean_surface_' + name] = (('theta', 'latitude'), geometry[name], {'units': NATIVE_UNITS[name]})
    for prefix, dim in [('iso_mean_first', 'pressure'), ('iso_mean_first_on_model', 'model_pressure')]:
        yy, yp, pp = [ds[prefix + '_' + n].values for n in ['D_yy', 'D_yp', 'D_pp']]
        det = yy * pp - yp * yp
        ds[prefix + '_psd_flag'] = ((dim, 'latitude'), np.where(np.isfinite(det), ((yy >= 0) & (pp >= 0) & (det >= -1e-12 * (abs(yy * pp) + yp * yp))).astype('int8'), -1),
                                    {'meaning': '1 PSD;0 fails;-1 missing. No masking or repair.'})
    ds.attrs.update(attrs)
    tmp = path.with_suffix('.partial.nc')
    ds.to_netcdf(tmp, engine='h5netcdf', encoding={n: {'zlib': True, 'complevel': 4} for n in ds.data_vars})
    tmp.replace(path)
    return {'sha256': sha256_file(path), 'bytes': path.stat().st_size, 'samples': mom_native.samples}


ATTRS = dict(
    isentropic_formalism='Instantaneous pressure from ARCO PS(Pa), hyam, hybm; instantaneous mass-weighted theta surfaces. Fields outside the selected vertical support remain undefined; no extrapolation.',
    theta_forcing='Q diagnosed on hourly trajectory endpoints. At the midpoint, hourly Eulerian Q is linearly interpolated and mass-centred with midpoint sigma.',
    initialization='Zero displacement at start; continuous through the interval and across restarts. The preceding hour supplies the initial Q.',
    pressure_mapping='iso_mean_first: mean native tensor and mean isentrope first, full J K J^T. iso_transform_first: hourly mapping, then time mean. *_on_model_* use the 81 fixed reference pressures A+B*101325 Pa.',
    geometry='D_pp = s^2 D_yy + 2 s r D_ytheta + r^2 D_thetatheta = D_pp_tilt + D_pp_mixed + D_pp_theta; D_yp = s D_yy + r D_ytheta = D_yp_tilt + D_yp_theta_cross; A_yp = r A_ytheta; K_yp = D_yp + A_yp; K_py = D_yp - A_yp.',
    masking='No time/history coverage cutoff, sign, SNR, amplitude or PSD rejection. Original physical crossing/zonal/geometric conditions remain.',
    statistics='Time means over finite hourly values (min_time_coverage = 0). No variability statistics.',
    equivalence='Same algorithm and floating-point operations as era5_monthly_diffusion_81.py --method isentropic --step-minutes 30; run --mode validate to confirm bitwise identity on this machine.')


# ----------------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------------
def report(out, phase, **details):
    record = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'phase': phase, 'peak_memory_MB': round(peak_memory_mb()), **details}
    print(json.dumps(record), flush=True)
    savejson(Path(out) / 'status.json', record)


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(a.results_dir) if a.results_dir else out
    results.mkdir(parents=True, exist_ok=True)
    final = results / result_name(a.start, a.end)
    config = settings(a)
    cfg, pref = make_config()
    script_hash = sha256_file(__file__)
    if (out / 'completion.json').exists():
        done = json.loads((out / 'completion.json').read_text())
        if done['configuration'] != config:
            raise ValueError('This output folder holds a different configuration; choose another folder')
        if not final.exists():
            raise FileNotFoundError(f'{out}/completion.json exists but {final} is missing; delete {out} to recompute')
        print('Already complete:', final)
        return
    if (out / 'settings.json').exists() and not a.resume:
        raise FileExistsError('A run exists in this folder. Use --resume or a fresh output folder.')
    set_num_threads(min(a.threads, numba_config.NUMBA_NUM_THREADS))
    slabs = Slabs(a.slab_threads)
    times = required_times(a)
    start = np.datetime64(a.start, 'h')
    report(out, 'opening_era5', snapshots=len(times))
    reader = Reader(times, a)
    timings = {}
    started = time.perf_counter()
    elapsed_prior = 0.

    def timed(name, tick):
        timings[name] = timings.get(name, 0.) + time.perf_counter() - tick

    try:
        if a.resume and (out / 'restart.pkl').exists():
            receipt = json.loads((out / 'restart.json').read_text())
            if sha256_file(out / 'restart.pkl') != receipt['sha256']:
                raise ValueError('restart.pkl does not match restart.json')
            with (out / 'restart.pkl').open('rb') as stream:
                restored = pickle.load(stream)
            if restored['configuration'] != config or restored['script_sha256'] != script_hash:
                raise ValueError('Restart configuration/script mismatch')
            done_hour = restored['hour']
            iso_state = restored['iso_state']
            mom_native, mom_tf = restored['moments']
            timings = restored['timings']
            elapsed_prior = restored['elapsed_seconds']
            report(out, 'resuming', hour=done_hour)
            prev = reader.frame(start + np.timedelta64(done_hour, 'h'))
            prior = remap(prev, cfg)
            prepare_means(prior, restored['prior_Q'], cfg, slabs)
            del restored
        else:
            savejson(out / 'settings.json', {'configuration': config, 'script_sha256': script_hash, 'threads': a.threads,
                                             'slab_threads': a.slab_threads, 'download_workers': a.download_workers,
                                             'batch_hours': a.batch_hours, 'prefetch_batches': a.prefetch_batches})
            prev = reader.frame(start)
            before = reader.frame(start - np.timedelta64(1, 'h'))
            report(out, 'initializing', note='first call compiles the Numba kernels (cached afterwards)')
            prior = remap(prev, cfg)
            prepare_means(prior, hourly_Q(before, prior, cfg), cfg, slabs)
            zero = np.zeros_like(prior['vp'])
            iso_state = (zero, zero.copy(), np.isfinite(prior['vp'] + prior['Qp'] + prior['sigma']).astype(np.float32))
            mom_native, mom_tf = Moments(), Moments()
            done_hour = 0
            del before
        check = slabs.pool is not None and not a.no_slab_check

        def checkpoint(hour):
            tick = time.perf_counter()
            payload = {'version': VERSION, 'script_sha256': script_hash, 'configuration': config, 'hour': hour,
                       'iso_state': iso_state, 'prior_Q': prior['Q'], 'moments': (mom_native, mom_tf),
                       'timings': timings, 'elapsed_seconds': elapsed_prior + time.perf_counter() - started}
            tmp = out / 'restart.pkl.partial'
            with tmp.open('wb') as stream:
                pickle.dump(payload, stream, protocol=5)
            tmp.replace(out / 'restart.pkl')
            savejson(out / 'restart.json', {'hour': hour, 'sha256': sha256_file(out / 'restart.pkl'), 'bytes': (out / 'restart.pkl').stat().st_size})
            timed('checkpoint', tick)

        report(out, 'integrating', first_hour=done_hour + 1, total_hours=config['hours'])
        for hour in range(done_hour + 1, config['hours'] + 1):
            hour_started = time.perf_counter()
            waited = reader.wait_seconds
            tick = time.perf_counter()
            now = reader.frame(start + np.timedelta64(hour, 'h'))
            waited = reader.wait_seconds - waited
            timings['read_wait'] = timings.get('read_wait', 0.) + waited
            timings['frame_build'] = timings.get('frame_build', 0.) + time.perf_counter() - tick - waited
            tick = time.perf_counter(); mid = midpoint(prev, now); timed('midpoint', tick)
            tick = time.perf_counter(); current = remap(now, cfg); timed('remap', tick)
            tick = time.perf_counter(); Q = hourly_Q(prev, current, cfg); timed('hourly_trajectory', tick)
            tick = time.perf_counter(); prepare_means(current, Q, cfg, slabs); timed('means', tick)
            if check:
                serial = {k: current[k] for k in ('p', 'u', 'v', 'omega', 'sigma')}
                prepare_means(serial, Q, cfg, SERIAL)
                if not all(same(current[k], serial[k]) for k in ('v_mean', 'Q_mean', 'vp', 'Qp')):
                    print('WARNING: slab means differ from serial on this NumPy build; continuing serially', flush=True)
                    current, slabs = serial, SERIAL
                del serial
            tick = time.perf_counter(); middle = remap(mid, cfg); timed('remap', tick)
            tick = time.perf_counter(); prepare_means(middle, .5 * (prior['Q'] + current['Q']), cfg, slabs); timed('means', tick)
            tick = time.perf_counter()
            iso_state = halfstep(prev, prior, middle, iso_state, cfg)
            iso_state = halfstep(mid, middle, current, iso_state, cfg)
            timed('halfstep_trajectories', tick)
            tick = time.perf_counter(); native = tensor_slabs(current, iso_state, cfg, slabs); timed('tensor', tick)
            if check:
                serial = tensor(current, *iso_state[:2], cfg, iso_state[2])
                if not all(same(native[k], serial[k]) for k in serial):
                    print('WARNING: slab tensor differs from serial on this NumPy build; continuing serially', flush=True)
                    native, slabs = serial, SERIAL
                report(out, 'slab_check', hour=hour, slabs_in_use=slabs.pool is not None)
                del serial
                check = False
            tick = time.perf_counter(); mapped = map_to_pressure(current, native, now.lat, cfg); timed('pressure_mapping', tick)
            tick = time.perf_counter()
            mom_native.add({k: native[k] for k in NATIVE_UNITS}, 3600.)
            mom_tf.add(mapped, 3600.)
            timed('statistics', tick)
            prior, prev = current, now
            del mid, middle, Q, native, mapped
            if hour % a.checkpoint_every == 0 or hour == config['hours']:
                checkpoint(hour)
            elapsed = elapsed_prior + time.perf_counter() - started
            st = reader.stats
            progress = {'hour': hour, 'total_hours': config['hours'], 'elapsed_seconds': round(elapsed, 1),
                        'seconds_this_hour': round(time.perf_counter() - hour_started, 2),
                        'eta_hours': round((config['hours'] - hour) * (time.perf_counter() - started) / (hour - done_hour) / 3600, 2),
                        'peak_memory_MB': round(peak_memory_mb()),
                        'download_MB_per_s': round(st['bytes'] / 1e6 / st['batch_seconds'], 1) if st['batch_seconds'] else None,
                        'timings_seconds': {k: round(v, 2) for k, v in timings.items()}}
            savejson(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if a.stop_after_hours and hour >= a.stop_after_hours and hour < config['hours']:
                if hour % a.checkpoint_every:
                    checkpoint(hour)
                report(out, 'stopped_with_checkpoint', hour=hour, note='continue with --resume and the same --start/--end')
                return
        report(out, 'writing_result')
        attrs = dict(ATTRS, title='ERA5 isentropic eddy-diffusion tensor (monthly means): ' + config['start'] + ' to ' + config['end'],
                     diagnostic_count=config['hours'], configuration_json=json.dumps(config), script_sha256=script_hash,
                     software=VERSION, input_cadence_seconds=3600, integration_step_seconds=1800)
        result = write_output(final, mom_native, mom_tf, prev.lat, cfg, pref, attrs)
        savejson(out / 'completion.json', {'completed': True, 'configuration': config, 'script_sha256': script_hash,
                                           'result_file': str(final), 'monthly': result,
                                           'elapsed_seconds': elapsed_prior + time.perf_counter() - started,
                                           'peak_memory_MB': peak_memory_mb(), 'timings_seconds': timings, 'reader': reader.stats})
        if not a.keep_restart:
            for name in ('restart.pkl', 'restart.json'):
                (out / name).unlink(missing_ok=True)
        report(out, 'complete', result=str(final))
    finally:
        reader.close()


# ----------------------------------------------------------------------------------
# Validation against the original script (side by side, same frames, bitwise)
# ----------------------------------------------------------------------------------
def validate(a):
    import importlib.util
    import xarray as xr
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('era5_reference', a.reference)
    ref = importlib.util.module_from_spec(spec)
    sys.modules['era5_reference'] = ref
    spec.loader.exec_module(ref)
    print('Compiling the original kernels (no cache, ~1 min) ...', flush=True)
    ref.load_science(a.threads)
    RC = ref.CORE
    set_num_threads(min(a.threads, numba_config.NUMBA_NUM_THREADS))
    slabs = Slabs(a.slab_threads)
    cfg, pref = make_config()
    _, ref_ph, _, _ = ref.MODEL.select_levels(1., 500.)
    rcfg = RC.Config(np.arange(300., 1000.01, 5.), np.unique(np.r_[ref.PRESSURE, ref_ph * 100]),
                     min_time_coverage=0, min_path_coverage=0, reset='never')
    results = {'reference_pressures_and_targets': same(ref_ph, pref) and same(rcfg.pressure, cfg.pressure)}
    start = np.datetime64(a.start, 'h')
    hours = a.validate_hours
    times = np.arange(start - np.timedelta64(1, 'h'), start + np.timedelta64(hours + 1, 'h'), np.timedelta64(1, 'h'))
    reader = SyntheticReader(times, a) if a.synthetic else Reader(times, a)
    readargs = RC.parser().parse_args(['--input', 'unused', '--output', 'unused', '--surface-pressure-unit', 'Pa', '--omega-unit', 'Pa/s'])

    def frames(t):
        raw = reader.raw(t)
        mine = build_frame(np.datetime64(t, 'h'), reader.lat, reader.lon, *raw, reader.aa, reader.bb)
        d = xr.Dataset({k: (('level', 'latitude', 'longitude'), x.copy(), {'units': ''}) for k, x in zip(['T', 'U', 'V', 'OMEGA'], raw[:4])},
                       coords={'level': LEVELS, 'latitude': reader.lat, 'longitude': reader.lon})
        d['PS'] = (('latitude', 'longitude'), raw[4].copy(), {'units': 'Pa'})
        d['hyam'] = ('level', reader.aa, {'units': 'Pa'})
        d['hybm'] = ('level', reader.bb, {'units': '1'})
        theirs = RC.snapshot_from_dataset(d, np.datetime64(t, 'ns'), readargs)
        ok = all(same(getattr(mine, n), getattr(theirs, n)) for n in ['p', 'theta', 'u', 'v', 'omega', 'lat', 'lon'])
        results['snapshot_' + str(np.datetime64(t, 'h'))] = ok
        return mine, theirs

    if not a.synthetic and not a.skip_reader_check and a.test_lon_stride != 4:
        results['reader_vs_original_StreamingReader'] = 'skipped (needs the operational 1-degree grid)'
    elif not a.synthetic and not a.skip_reader_check:
        try:
            if a.model_url != MODEL_URL or a.surface_url != SURFACE_URL:
                ref.MODEL_URL, ref.SURFACE_URL = a.model_url, a.surface_url   # local mirror (testing)
            ns = types.SimpleNamespace(start=str(start), end=str(start + np.timedelta64(1, 'h')), method='isentropic',
                                       batch_hours=2, download_workers=8, output=str(out), input_mode='stream')
            sr = ref.StreamingReader(ns)
            ok = True
            for t in times[:2]:
                theirs = sr.frame(np.datetime64(t, 'ns'), {})
                mine = reader.frame(t)
                ok &= all(same(getattr(mine, n), getattr(theirs, n)) for n in ['p', 'theta', 'u', 'v', 'omega'])
            sr.close()
            results['reader_vs_original_StreamingReader'] = ok
        except Exception as exc:
            results['reader_vs_original_StreamingReader'] = f'skipped ({type(exc).__name__}: {exc})'
        print('Reader check:', results['reader_vs_original_StreamingReader'], flush=True)

    t_mine = t_ref = 0.
    # Initialization (run_locked, not resuming)
    prev, r_prev = frames(start)
    before, r_before = frames(start - np.timedelta64(1, 'h'))
    tick = time.perf_counter()
    r_prior = RC.remap(r_prev, rcfg)
    r_dep = RC.departures(r_before, r_prev, r_prior, 3600., rcfg)
    RC.prepare_means(r_prior, r_dep['Q'], rcfg)
    r_zero = np.zeros_like(r_prior['vp'])
    r_state = (r_zero, r_zero.copy(), np.isfinite(r_prior['vp'] + r_prior['Qp'] + r_prior['sigma']).astype(np.float32))
    t_ref += time.perf_counter() - tick
    del r_dep
    tick = time.perf_counter()
    prior = remap(prev, cfg)
    prepare_means(prior, hourly_Q(before, prior, cfg), cfg, slabs)
    zero = np.zeros_like(prior['vp'])
    iso_state = (zero, zero.copy(), np.isfinite(prior['vp'] + prior['Qp'] + prior['sigma']).astype(np.float32))
    t_mine += time.perf_counter() - tick
    results['initial_Q'] = same(prior['Q'], r_prior['Q']) and same(prior['vp'], r_prior['vp']) and same(iso_state[2], r_state[2])
    mom = (Moments(), Moments())
    r_mom = (RC.Moments(), RC.Moments())
    for hour in range(1, hours + 1):
        now, r_now = frames(start + np.timedelta64(hour, 'h'))
        tick = time.perf_counter()
        r_mid = RC.Snapshot(r_prev.time + np.timedelta64(30, 'm'), r_prev.lat, r_prev.lon,
                            *[(getattr(r_prev, n) + getattr(r_now, n)) * .5 for n in ['p', 'theta', 'u', 'v', 'omega']])
        r_current = RC.remap(r_now, rcfg)
        r_dep = RC.departures(r_prev, r_now, r_current, 3600., rcfg)
        RC.prepare_means(r_current, r_dep['Q'], rcfg)
        del r_dep
        r_middle = RC.remap(r_mid, rcfg)
        RC.prepare_means(r_middle, .5 * (r_prior['Q'] + r_current['Q']), rcfg)
        for oldframe, newframe, oldstate, newstate in [(r_prev, r_mid, r_prior, r_middle), (r_mid, r_now, r_middle, r_current)]:
            r_dep = RC.departures(oldframe, newframe, newstate, 1800., rcfg)
            r_state = RC.advance_displacements(oldstate, newstate, *r_state, r_dep, r_now.lat, r_now.lon, 1800., rcfg)
            del r_dep
        r_native = RC.tensor(r_current, *r_state[:2], rcfg, r_state[2])
        r_mapped = RC.map_to_pressure(r_current, r_native, r_now.lat, rcfg)
        r_mom[0].add({k: r_native[k] for k in NATIVE_UNITS}, 3600.)
        r_mom[1].add(r_mapped, 3600.)
        t_ref += time.perf_counter() - tick
        tick = time.perf_counter()
        mid = midpoint(prev, now)
        current = remap(now, cfg)
        prepare_means(current, hourly_Q(prev, current, cfg), cfg, slabs)
        middle = remap(mid, cfg)
        prepare_means(middle, .5 * (prior['Q'] + current['Q']), cfg, slabs)
        iso_state = halfstep(prev, prior, middle, iso_state, cfg)
        iso_state = halfstep(mid, middle, current, iso_state, cfg)
        native = tensor_slabs(current, iso_state, cfg, slabs)
        mapped = map_to_pressure(current, native, now.lat, cfg)
        mom[0].add({k: native[k] for k in NATIVE_UNITS}, 3600.)
        mom[1].add(mapped, 3600.)
        t_mine += time.perf_counter() - tick
        checks = {'Q': same(current['Q'], r_current['Q']), 'eta': same(iso_state[0], r_state[0]),
                  'xi': same(iso_state[1], r_state[1]), 'support': same(iso_state[2], r_state[2]),
                  'native_tensor': all(same(native[k], r_native[k]) for k in NATIVE_UNITS),
                  'pressure_mapped': all(same(mapped[k], r_mapped[k]) for k in r_mapped)}
        results[f'hour_{hour}'] = checks
        print(f'hour {hour}:', checks, flush=True)
        prior, prev, r_prior, r_prev = current, now, r_current, r_now
    if hours == 0:
        mom = None
    mf, _ = mean_first(_means(mom[0]), reader.lat, cfg) if mom else ({}, None)
    if mom:
        rm = _means(r_mom[0])
        r_native = {k: rm[k].copy() for k in ['D_yy', 'D_ytheta', 'D_thetatheta', 'A_ytheta']}
        r_mf = RC.map_to_pressure({'p': rm['p_mean'][..., None], 'sigma': rm['sigma_mean'][..., None], 'v_mean': rm['v_mass_mean'],
                                   'Q_mean': rm['Q_mass_mean']}, r_native, reader.lat, rcfg)
        results['mean_first_tensor_and_geometry'] = all(same(mf[k], r_mf[k]) for k in r_mf)
        tf, r_tf = _means(mom[1]), _means(r_mom[1])
        results['transform_first_means'] = all(same(tf[k], r_tf[k]) for k in r_tf)
    else:
        results['mean_first_tensor_and_geometry'] = results['transform_first_means'] = 'not run (0 hours)'
    flat = [v for v in results.values() if not isinstance(v, str)]
    flat = [x for v in flat for x in (v.values() if isinstance(v, dict) else [v])]
    results['ALL_BITWISE_IDENTICAL'] = bool(all(flat))
    results['seconds_original_pipeline'] = round(t_ref, 1)
    results['seconds_fast_pipeline_incl_first_compile'] = round(t_mine, 1)
    savejson(out / 'validation.json', results)
    reader.close()
    print(json.dumps({k: results[k] for k in ['ALL_BITWISE_IDENTICAL', 'mean_first_tensor_and_geometry', 'transform_first_means',
                                              'seconds_original_pipeline', 'seconds_fast_pipeline_incl_first_compile']}, indent=1))
    if not results['ALL_BITWISE_IDENTICAL']:
        raise SystemExit('VALIDATION FAILED: see ' + str(out / 'validation.json'))
    print('VALIDATION PASSED: bitwise identical to the original for every compared array.')


# ----------------------------------------------------------------------------------
# Year: months in sequence, each in a fresh process (no memory carry-over), skip/resume/retry
# ----------------------------------------------------------------------------------
def parse_months(text):
    months = set()
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        months.update(range(int(lo), int(hi or lo) + 1))
    if not months or min(months) < 1 or max(months) > 12:
        raise ValueError('--months takes values 1-12, e.g. 1-12 or 2,5-7')
    return sorted(months)


def year(a):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    passthrough = ['--threads', str(a.threads), '--slab-threads', str(a.slab_threads),
                   '--download-workers', str(a.download_workers), '--batch-hours', str(a.batch_hours),
                   '--prefetch-batches', str(a.prefetch_batches), '--checkpoint-every', str(a.checkpoint_every),
                   '--model-url', a.model_url, '--surface-url', a.surface_url, '--test-lon-stride', str(a.test_lon_stride)]
    for flag in ('verify_md5', 'no_slab_check', 'keep_restart'):
        if getattr(a, flag):
            passthrough.append('--' + flag.replace('_', '-'))
    if a.stop_after_hours:
        passthrough += ['--stop-after-hours', str(a.stop_after_hours)]
    for month in parse_months(a.months):
        start, end = month_bounds(a.year, month)
        if a.test_max_hours:
            end = start + np.timedelta64(a.test_max_hours, 'h')
        final = root / result_name(start, end)
        if final.exists():
            print(json.dumps({'month': f'{a.year}-{month:02d}', 'status': 'exists, skipped', 'file': str(final)}), flush=True)
            continue
        work = root / 'work' / f'{a.year:04d}_{month:02d}'
        command = [sys.executable, '-u', script, '--mode', 'run', '--start', str(start), '--end', str(end),
                   '--output', str(work), '--results-dir', str(root), '--resume'] + passthrough
        for attempt in range(1, a.retries + 2):
            began = time.time()
            print(json.dumps({'month': f'{a.year}-{month:02d}', 'status': 'running', 'attempt': attempt,
                              'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}), flush=True)
            code = subprocess.call(command)
            if code == 0 and final.exists():
                print(json.dumps({'month': f'{a.year}-{month:02d}', 'status': 'done', 'file': str(final),
                                  'hours_this_attempt': round((time.time() - began) / 3600, 2)}), flush=True)
                break
            print(json.dumps({'month': f'{a.year}-{month:02d}', 'status': 'interrupted', 'attempt': attempt, 'exit_code': code,
                              'note': f'resuming from the last restart file in {a.retry_delay} s'}), flush=True)
            time.sleep(a.retry_delay)
        else:
            raise SystemExit(f'{final.name} did not complete after {a.retries + 1} attempts; see the log and {work}/status.json. '
                             'Rerun the same command to continue from the last restart file.')
    print(json.dumps({'year': a.year, 'status': 'all requested months available', 'folder': str(root)}), flush=True)


# ----------------------------------------------------------------------------------
def plan(a):
    config = settings(a)
    times = required_times(a)
    chunks = len(times) * (len(FIELDS) * 6 + 1)
    batch_gb = a.batch_hours * (len(FIELDS) * len(LEVELS) + 1) * 361 * (1440 // a.test_lon_stride) * 4 / 2 ** 30
    print(json.dumps({'configuration': config, 'result_file': result_name(a.start, a.end), 'snapshots': len(times), 'zarr_chunk_reads': chunks,
                      'decoded_TB': round(len(times) * len(FIELDS) * 6 * CHUNK_DECODED_BYTES / 1e12, 3),
                      'batch_GiB': round(batch_gb, 2), 'batches_in_memory': 1 + a.prefetch_batches,
                      'numba_threads': a.threads, 'slab_threads': a.slab_threads, 'download_workers': a.download_workers,
                      'note': 'Read throughput >= ~220 MB/s (decoded) keeps the download hidden behind compute on a c3-standard-8.'}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['plan', 'run', 'year', 'validate'], default='plan')
    p.add_argument('--year', type=int, default=2010, help='year mode: calendar year')
    p.add_argument('--months', default='1-12', help='year mode: months, e.g. 1-12 or 2,5-7')
    p.add_argument('--retries', type=int, default=3, help='year mode: automatic resumes per month after a crash')
    p.add_argument('--results-dir', help='run mode: folder for the final NetCDF (default: --output)')
    p.add_argument('--keep-restart', action='store_true', help='keep restart.pkl after a month completes')
    p.add_argument('--start', default='2010-01-01T00', help='inclusive, whole UTC hour')
    p.add_argument('--end', default='2010-02-01T00', help='exclusive end of the integration, whole UTC hour')
    p.add_argument('--output', default='era5_iso_fast_out', help='run: work folder; year: results folder')
    p.add_argument('--resume', action='store_true', help='continue from <output>/restart.pkl')
    p.add_argument('--threads', type=int, default=8, help='Numba threads (c3-standard-8: 8; try 4 to compare)')
    p.add_argument('--slab-threads', type=int, default=4, help='thread pool for the NumPy reductions')
    p.add_argument('--download-workers', type=int, default=8, help='parallel Zarr chunk requests')
    p.add_argument('--batch-hours', type=int, default=4, help='hourly snapshots per download batch')
    p.add_argument('--prefetch-batches', type=int, default=1, help='batches downloaded ahead of the integration')
    p.add_argument('--checkpoint-every', type=int, default=24, help='simulated hours between restart files')
    p.add_argument('--stop-after-hours', type=int, default=0, help='stop (with a restart file) after this simulated hour')
    p.add_argument('--verify-md5', action='store_true', help='check GCS md5 of every chunk (extra CPU)')
    p.add_argument('--no-slab-check', action='store_true', help='skip the first-hour slab-vs-serial exactness check')
    p.add_argument('--reference', help='validate: path to era5_monthly_diffusion_81.py')
    p.add_argument('--validate-hours', type=int, default=3)
    p.add_argument('--synthetic', action='store_true', help='validate on synthetic fields (no download)')
    p.add_argument('--skip-reader-check', action='store_true', help='validate: skip the comparison with the original dask/zarr reader')
    p.add_argument('--model-url', default=MODEL_URL, help=argparse.SUPPRESS)
    p.add_argument('--surface-url', default=SURFACE_URL, help=argparse.SUPPRESS)
    p.add_argument('--test-lon-stride', type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument('--test-max-hours', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--retry-delay', type=int, default=60, help=argparse.SUPPRESS)
    return p


def main():
    a = parser().parse_args()
    if a.mode == 'plan':
        plan(a)
    elif a.mode == 'run':
        run(a)
    elif a.mode == 'year':
        year(a)
    elif a.mode == 'validate':
        if not a.reference:
            raise SystemExit('--mode validate needs --reference path/to/era5_monthly_diffusion_81.py')
        validate(a)


if __name__ == '__main__':
    main()
