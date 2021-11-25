#!/usr/bin/env python
from typing import Tuple

import cv2
import numpy as np

from stereo_rectification.core import Array

from .projective_transform import estimate_projective_transform
from .shearing_transform import estimate_shearing_transform
from .similarity_transform import estimate_similarity_transform

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["stereo_rectify_uncalibrated"]


def get_additional_scale_translation(
    img_size: Tuple[int, int],
    H_left: Array[Tuple[Literal[3], Literal[3]], np.float64],
    H_right: Array[Tuple[Literal[3], Literal[3]], np.float64],
) -> Tuple[Array[Tuple[Literal[3], Literal[3]], np.float64], Array[Tuple[Literal[3], Literal[3]], np.float64]]:
    w, h = img_size
    mid_points = np.array(
        [(w - 1) / 2.0, 0, (w - 1), (h - 1) / 2.0, (w - 1) / 2.0, (h - 1), 0, (h - 1) / 2.0], dtype=np.float64
    ).reshape(-1, 2)
    mid_points_lprojected = cv2.perspectiveTransform(mid_points.reshape(1, -1, 2), H_left).reshape(-1, 2)
    mid_points_rprojected = cv2.perspectiveTransform(mid_points.reshape(1, -1, 2), H_right).reshape(-1, 2)

    def get_dimensions(mid_points):
        w_dimensions = cv2.norm(mid_points_lprojected[1] - mid_points_lprojected[3])
        h_dimensions = cv2.norm(mid_points_lprojected[0] - mid_points_lprojected[2])
        return w_dimensions, h_dimensions

    w_l, h_l = get_dimensions(mid_points_lprojected)
    w_r, h_r = get_dimensions(mid_points_rprojected)
    scale_w = w / max(w_l, w_r)
    scale_h = h / max(h_l, h_r)
    H_scale = np.eye(3, 3, dtype=np.float64)
    H_scale[0, 0] = scale_w
    H_scale[1, 1] = scale_h

    # flip left right
    if mid_points_lprojected[0][0] > mid_points_lprojected[1][0]:
        H_scale[0, 0] *= -1

    # flip up down
    if mid_points_lprojected[0][1] > mid_points_lprojected[3][1]:
        H_scale[1, 1] *= -1

    mid_points_lprojected = cv2.perspectiveTransform(mid_points_lprojected.reshape(1, -1, 2), H_scale).reshape(-1, 2)
    mid_points_rprojected = cv2.perspectiveTransform(mid_points_rprojected.reshape(1, -1, 2), H_scale).reshape(-1, 2)
    mid_points_lprojected_avg = np.average(mid_points_lprojected, axis=0)
    mid_points_rprojected_avg = np.average(mid_points_rprojected, axis=0)
    y_translation = (mid_points_lprojected_avg[1] + mid_points_rprojected_avg[1]) / 2 - h / 2.0
    x_translation_left = mid_points_lprojected_avg[0] - w / 2.0
    x_translation_right = mid_points_rprojected_avg[0] - w / 2.0

    H_translation_left = np.eye(3, 3, dtype=np.float64)
    H_translation_right = np.eye(3, 3, dtype=np.float64)
    H_translation_left[0, 2] = -x_translation_left
    H_translation_right[0, 2] = -x_translation_right
    H_translation_left[1, 2] = -y_translation
    H_translation_right[1, 2] = -y_translation

    return H_translation_left.dot(H_scale), H_translation_right.dot(H_scale)


def stereo_rectify_uncalibrated(F: Array[Tuple[Literal[3], Literal[3]], np.float64], img_size: Tuple[int, int]):
    H_left, H_right = estimate_projective_transform(F, img_size)
    Hr_left, Hr_right = estimate_similarity_transform(F, img_size, H_left, H_right)

    H_left = Hr_left.dot(H_left)
    H_right = Hr_right.dot(H_right)

    Hs_left, Hs_right = estimate_shearing_transform(F, img_size, H_left, H_right)
    H_left = Hs_left.dot(H_left)
    H_right = Hs_right.dot(H_right)

    H_additional_left, H_additional_right = get_additional_scale_translation(img_size, H_left, H_right)
    H_left = H_additional_left.dot(H_left)
    H_right = H_additional_right.dot(H_right)

    return H_left, H_right
