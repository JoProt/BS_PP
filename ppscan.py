#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Palmprint Scanner
    =================
    [your ad here]
    
    :license: who knows
    :format: black, reST docstrings
"""

import os
import sys
import numpy as np
import cv2 as cv


# # # # # # # # #
# Preprocessing #
# # # # # # # # #


def preprocess(img):
    """
    Wende Weichzeichner an und wandle in Binärbild um.
    :param img: Eingangsbild
    :returns: Binärbild
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsv, lower, upper)
    blurred = cv.GaussianBlur(skinRegionHSV, (5, 5), 0)
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    return thresh


def get_contours(mask_img):
    contours, hierarchy = cv.findContours(
        mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    return contours, hull


def get_defects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects


def find_fingers(img):
    """
    Finde die "Täler" zwischen Zeige- und Mittelfinger bzw
    Ring- und kleinem Finger.
    :param img: vorverarbeitetes Bild
    :returns: 2 Koordinaten-Tupel
    """
    mask_img = make_binary(img)
    contours, hull = get_contours(mask_img)
    finger_points = []
    defects = get_defects(contours)
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(contours[s][0])
            end = tuple(contours[e][0])
            far = tuple(contours[f][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            if angle <= np.pi / 2:
                finger_points.append(far)

    # nur interessante Punkte zurückgeben
    return finger_points[1], finger_points[3]


def fit():
    return


# ...


def main():
    print("it werks")
    return


if __name__ == "__main__":
    sys.exit(main())
