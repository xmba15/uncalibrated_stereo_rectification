#!/usr/bin/env python
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .types import Array

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["match_features", "find_fundamental_matrix", "draw_epi_lines", "estimate_epipoles", "skew", "normalize"]


DETECTOR_NORMS_DICT = {
    "SIFT": (cv2.SIFT_create(), cv2.NORM_L2),
    "ORB": (cv2.ORB_create(), cv2.NORM_HAMMING),
    "AKAZE": (cv2.AKAZE_create(), cv2.NORM_HAMMING),
    "BRISK": (cv2.BRISK_create(), cv2.NORM_HAMMING),
}
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6


def _init_detector_matcher(detector_name: str) -> Tuple[cv2.Feature2D, cv2.DescriptorMatcher]:
    try:
        detector, norm = DETECTOR_NORMS_DICT[detector_name]
    except KeyError:
        detector, norm = DETECTOR_NORMS_DICT["ORB"]

    flann_params = (
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        if norm == cv2.NORM_L2
        else dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    )
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    return detector, matcher


def match_features(
    img1: Array[Tuple[int, int], np.uint8],
    img2: Array[Tuple[int, int], np.uint8],
    detector_name: str = "ORB",
    ratio: float = 0.6,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    assert img1.ndim == 2 and img1.dtype == np.uint8, "img1 is invalid"
    assert img2.ndim == 2 and img2.dtype == np.uint8, "img2 is invalid"

    keypoint_detector, keypoint_matcher = _init_detector_matcher(detector_name)

    kps1, des1 = keypoint_detector.detectAndCompute(img1, None)
    kps2, des2 = keypoint_detector.detectAndCompute(img2, None)
    matches = keypoint_matcher.knnMatch(des1, des2, k=2)

    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < ratio * m[1].distance]

    return kps1, kps2, matches


def find_fundamental_matrix(
    img1: Array[Tuple[int, int], np.uint8],
    img2: Array[Tuple[int, int], np.uint8],
    detector_name: str = "ORB",
    ratio: float = 0.6,
) -> Tuple[
    Optional[Array[Tuple[Literal[3], Literal[3]], np.float64]],
    Array[Tuple[int, Literal[2]], np.float64],
    Array[Tuple[int, Literal[2]], np.float64],
]:
    all_kps1, all_kps2, matches = match_features(img1, img2, detector_name, ratio)
    kps1 = np.asarray([all_kps1[m.queryIdx].pt for m in matches])
    kps2 = np.asarray([all_kps2[m.trainIdx].pt for m in matches])

    num_keypoints = len(matches)
    if num_keypoints < 7:
        return None, kps1, kps2

    flag = cv2.FM_7POINT if num_keypoints == 7 else cv2.FM_8POINT
    F, mask = cv2.findFundamentalMat(kps1, kps2, flag)

    # get inlier keypoints
    kps1 = kps1[mask.ravel() == 1]
    kps2 = kps2[mask.ravel() == 1]

    return F, kps1, kps2


def draw_epi_lines(
    img1: Array[Tuple[int, ...], np.uint8],
    img2: Array[Tuple[int, ...], np.uint8],
    F: Array[Tuple[Literal[3], Literal[3]], np.float64],
    kps1: Array[Tuple[int, Literal[2]], np.float64],
    kps2: Array[Tuple[int, Literal[2]], np.float64],
    seed: int = 2021,
) -> Tuple[Array[Tuple[int, ...], np.uint8], Array[Tuple[int, ...], np.uint8]]:
    np.random.seed(seed)

    visualized1 = img1.copy()
    visualized2 = img2.copy()

    lines1 = cv2.computeCorrespondEpilines(kps2, 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(kps1, 1, F).reshape(-1, 3)

    width = img1.shape[1]

    def draw_point_line(img, point, line, color):
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [width, -(line[2] + line[0] * width) / line[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img, tuple(map(int, point)), 5, color, -1)

    for (kp1, kp2, line1, line2) in zip(kps1, kps2, lines1, lines2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw_point_line(visualized1, kp1, line1, color)
        draw_point_line(visualized2, kp2, line2, color)

    return visualized1, visualized2


def estimate_epipoles(
    F: Array[Tuple[Literal[3], Literal[3]], np.float64]
) -> Tuple[Array[Tuple[Literal[3]], np.float64], Array[Tuple[Literal[3]], np.float64]]:
    _, U, Vt = cv2.SVDecomp(F)
    left_epipole = Vt[2, :] / Vt[2, 2]
    right_epipole = U[:, 2] / U[2, 2]

    return (left_epipole, right_epipole)


def skew(v: Array[Tuple[Literal[3]], Any]) -> Array[Tuple[Literal[3], Literal[3]], Any]:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def normalize(v: Array[Tuple[int, ...], Any]) -> Array[Tuple[int, ...], Any]:
    norm = np.linalg.norm(v)
    return v if np.isclose(norm, 0) else v / norm
