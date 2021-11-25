#!/usr/bin/env python
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares

from stereo_rectification.core import Array, estimate_epipoles, normalize, skew

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["estimate_projective_transform"]


def _get_initial_guess(
    A: Array[Tuple[Literal[2], Literal[2]], np.float64], B: Array[Tuple[Literal[2], Literal[2]], np.float64]
) -> Array[Tuple[Literal[2]], np.float64]:
    D = np.linalg.cholesky(A)
    D_inv = np.linalg.inv(D)
    DBD = D_inv.transpose().dot(B).dot(D_inv)
    _, v = np.linalg.eig(DBD)
    return D_inv.dot(v[:, 0])


def estimate_projective_transform(
    F: Array[Tuple[Literal[3], Literal[3]], np.float64], img_size: Tuple[int, int]
) -> Tuple[Array[Tuple[Literal[3], Literal[3]], np.float64], Array[Tuple[Literal[3], Literal[3]], np.float64]]:
    left_epipole, _ = estimate_epipoles(F)

    w, h = img_size
    PPt = np.zeros((3, 3), dtype=np.float64)
    PPt[0, 0] = w ** 2 - 1
    PPt[1, 1] = h ** 2 - 1
    PPt *= (w * h) / 12.0

    PcPct = np.array(
        [
            [(w - 1) ** 2, (w - 1) * (h - 1), 2 * (w - 1)],
            [(w - 1) * (h - 1), (h - 1) ** 2, 2 * (h - 1)],
            [2 * (w - 1), 2 * (h - 1), 4],
        ],
        dtype=np.float64,
    )
    PcPct /= 4

    e_skew = skew(left_epipole)
    e_skew_t = e_skew.transpose()

    A = e_skew_t.dot(PPt).dot(e_skew)[:2, :2]
    B = e_skew_t.dot(PcPct).dot(e_skew)[:2, :2]
    A_prime = F.transpose().dot(PPt).dot(F)[:2, :2]
    B_prime = F.transpose().dot(PcPct).dot(F)[:2, :2]

    z1 = _get_initial_guess(A, B)
    z2 = _get_initial_guess(A_prime, B_prime)
    z = (normalize(z1) + normalize(z2)) / 2.0
    z /= z[1]

    def _objective_func(coeffs):
        x = np.array([coeffs, 1], dtype=np.float64).reshape(-1, 1)
        x_t = x.transpose()
        return x_t.dot(A).dot(x)[0] / x_t.dot(B).dot(x)[0] + x_t.dot(A_prime).dot(x)[0] / x_t.dot(B_prime).dot(x)[0]

    res = least_squares(_objective_func, z[0])

    z = np.array([res.x, 1, 0], dtype=np.float64)

    wv = e_skew.dot(z)
    wv /= wv[2]
    wv_prime = F.dot(z)
    wv_prime /= wv_prime[2]

    Hp_left = np.eye(3, 3, dtype=np.float64)
    Hp_left[2, :] = wv
    Hp_right = np.eye(3, 3, dtype=np.float64)
    Hp_right[2, :] = wv_prime

    return (Hp_left, Hp_right)
