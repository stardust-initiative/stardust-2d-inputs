"""Same Mie hemisphere integrals via finite Legendre phase expansion.
P integrates to 2. If P=sum a_l P_l, hemisphere(beta_mu)=
.5 sum a_l P_l(-mu) J_l, J_l=int_0^1 P_l(u)du.
The legacy beta_bar definition is the uniform-mu mean: .5 sum a_l(-1)^l J_l².
No clipping or normalization repair is performed.
"""

from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre, eval_legendre
import miepython

DEGREE = 384


@lru_cache(maxsize=8)
def kernel(nquad, mu_tuple):
    if nquad <= DEGREE:
        raise ValueError("Quadrature order must exceed Legendre degree 384")
    nodes, w = roots_legendre(nquad)
    degree_index = np.arange(DEGREE + 1)
    V = np.polynomial.legendre.legvander(nodes, DEGREE)
    projection = ((2 * degree_index + 1) / 2)[:, None] * V.T * w[None, :]
    J = np.zeros(DEGREE + 1)
    J[0] = 1
    J[1:] = (
        eval_legendre(degree_index[1:] - 1, 0.0)
        - eval_legendre(degree_index[1:] + 1, 0.0)
    ) / (2 * degree_index[1:] + 1)
    mus = np.array((1.0,) + mu_tuple)
    H = 0.5 * np.polynomial.legendre.legvander(-mus, DEGREE) * J[None, :]
    B = 0.5 * (-1.0) ** degree_index * J**2
    return nodes, H @ projection, B @ projection, w / 2


def beta_calc(m, x, ntt=512, mu_vect=None):
    if not np.isfinite(x) or not 0 < x < 140:
        raise ValueError(
            "Size parameter must be between 0 and 140; requalify larger sizes"
        )
    mus = np.asarray(mu_vect, dtype=float) if mu_vect is not None else np.array([])
    if mus.ndim != 1 or not np.all(np.isfinite(mus)) or np.any((mus < 0) | (mus > 1)):
        raise ValueError(
            "Incident cosines must be a finite one-dimensional array in [0, 1]"
        )
    mu_tuple = tuple(mus)
    nodes, H, B, norm = kernel(ntt, mu_tuple)
    P = miepython.i_unpolarized(np.conjugate(m), x, nodes, "4pi")
    err = abs(float(P @ norm) - 1.0)
    if not np.isfinite(err) or err >= 1e-5:
        raise ValueError(f"Phase normalization failed: x={x}, error={err}")
    vals = H @ P
    bar = float(B @ P)
    return float(vals[0]), bar, vals[1:] if mu_vect is not None else None
