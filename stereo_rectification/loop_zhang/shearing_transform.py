#!/usr/bin/env python
from typing import Tuple

import cv2
import numpy as np

from stereo_rectification.core import Array

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["estimate_shearing_transform"]


def estimate_shearing_transform_single_camera(
    F: Array[Tuple[Literal[3], Literal[3]], np.float64],
    img_size: Tuple[int, int],
    H: Array[Tuple[Literal[3], Literal[3]], np.float64],
) -> Array[Tuple[Literal[3], Literal[3]], np.float64]:

    w, h = img_size
    mid_points = np.array(
        [(w - 1) / 2.0, 0, (w - 1), (h - 1) / 2.0, (w - 1) / 2.0, (h - 1), 0, (h - 1) / 2.0], dtype=np.float64
    )
    mid_points_projected = cv2.perspectiveTransform(mid_points.reshape(1, -1, 2), H).reshape(-1, 2)

    x = mid_points_projected[1, :] - mid_points_projected[3, :]
    y = mid_points_projected[2, :] - mid_points_projected[0, :]

    d = h * w * (x[1] * y[0] - x[0] * y[1])
    a = (h ** 2 * x[1] ** 2 + w ** 2 * y[1] ** 2) / d
    b = -(h ** 2 * x[0] * x[1] + w ** 2 * y[0] * y[1]) / d

    Hs = np.eye(3, 3, dtype=np.float64)
    Hs[0, :2] = [a, b]

    return Hs


def estimate_shearing_transform(
    F: Array[Tuple[Literal[3], Literal[3]], np.float64],
    img_size: Tuple[int, int],
    H_left: Array[Tuple[Literal[3], Literal[3]], np.float64],
    H_right: Array[Tuple[Literal[3], Literal[3]], np.float64],
) -> Tuple[Array[Tuple[Literal[3], Literal[3]], np.float64], Array[Tuple[Literal[3], Literal[3]], np.float64]]:
    Hs_left = estimate_shearing_transform_single_camera(F, img_size, H_left)
    Hs_right = estimate_shearing_transform_single_camera(F, img_size, H_right)

    return (Hs_left, Hs_right)
