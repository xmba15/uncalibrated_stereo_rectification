#!/usr/bin/env python
import argparse
import os
import sys

import cv2

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
if True:  # noqa: E402
    from stereo_rectification.core import draw_epi_lines, find_fundamental_matrix
    from stereo_rectification.loop_zhang import (
        stereo_rectify_uncalibrated as stereo_rectify_uncalibrated_lz,
    )


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--left_image_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--right_image_path",
        type=str,
        required=True,
    )

    return parser.parse_args()


def rectify_with_opencv(left_image, right_image, F, kps1, kps2):
    img_size = (left_image.shape[1], left_image.shape[0])
    _, H1, H2 = cv2.stereoRectifyUncalibrated(kps1, kps2, F, img_size)
    H1 /= H1[2, 2]
    H2 /= H2[2, 2]

    left_rectified = cv2.warpPerspective(left_image, H1, img_size)
    right_rectified = cv2.warpPerspective(right_image, H2, img_size)
    opencv_rectified = cv2.hconcat([left_rectified, right_rectified])

    return opencv_rectified


def main():
    args = get_args()
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)
    assert left_image is not None, "failed to load {}".format(args.left_image_path)
    assert right_image is not None, "failed to load {}".format(args.right_image_path)
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    F, kps1, kps2 = find_fundamental_matrix(left_image_gray, right_image_gray, "ORB")

    if len(kps1) < 7:
        print("not enough key points")
        sys.exit(-1)

    left_image, right_image = draw_epi_lines(left_image, right_image, F, kps1, kps2)
    not_rectified = cv2.hconcat([left_image, right_image])

    opencv_rectified = rectify_with_opencv(left_image, right_image, F, kps1, kps2)

    img_size = (left_image.shape[1], left_image.shape[0])
    H1_lz, H2_lz = stereo_rectify_uncalibrated_lz(F, img_size)

    left_rectified_lz = cv2.warpPerspective(left_image, H1_lz, img_size)
    right_rectified_lz = cv2.warpPerspective(right_image, H2_lz, img_size)
    lz_rectified = cv2.hconcat([left_rectified_lz, right_rectified_lz])

    [cv2.namedWindow(name, cv2.WINDOW_NORMAL) for name in ("not_rectified", "opencv_rectified", "lz_rectified")]
    cv2.imshow("not_rectified", not_rectified)
    cv2.imshow("opencv_rectified", opencv_rectified)
    cv2.imshow("lz_rectified", lz_rectified)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
