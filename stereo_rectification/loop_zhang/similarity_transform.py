#!/usr/bin/env python
from typing import Tuple

import cv2
import numpy as np

from stereo_rectification.core import Array

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["estimate_similarity_transform"]


def estimate_similarity_transform(
    F: Array[Tuple[Literal[3], Literal[3]], np.float64],
    img_size: Tuple[int, int],
    H_left: Array[Tuple[Literal[3], Literal[3]], np.float64],
    H_right: Array[Tuple[Literal[3], Literal[3]], np.float64],
) -> Tuple[Array[Tuple[Literal[3], Literal[3]], np.float64], Array[Tuple[Literal[3], Literal[3]], np.float64]]:
    wv = H_left[2, :]
    wv_prime = H_right[2, :]

    w, h = img_size
    corner_points = np.array([0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1], dtype=np.float64).reshape(-1, 2)
    corner_points_lprojected = cv2.perspectiveTransform(corner_points.reshape(1, -1, 2), H_left).reshape(-1, 2)
    corner_points_rprojected = cv2.perspectiveTransform(corner_points.reshape(1, -1, 2), H_right).reshape(-1, 2)
    min_y_l = min(corner_points_lprojected[:, 1])
    min_y_r = min(corner_points_rprojected[:, 1])

    translation_term = -min(min_y_l, min_y_r)

    Hr_left = np.array(
        [
            [F[2, 1] - wv[1] * F[2, 2], wv[0] * F[2, 2] - F[2, 0], 0],
            [F[2, 0] - wv[0] * F[2, 2], F[2, 1] - wv[1] * F[2, 2], F[2, 2] + translation_term],
            [0, 0, 1],
        ],
    )

    Hr_right = np.array(
        [
            [wv_prime[1] * F[2, 2] - F[1, 2], F[0, 2] - wv_prime[0] * F[2, 2], 0],
            [wv_prime[0] * F[2, 2] - F[0, 2], wv_prime[1] * F[2, 2] - F[1, 2], translation_term],
            [0, 0, 1],
        ],
    )

    return (Hr_left, Hr_right)
