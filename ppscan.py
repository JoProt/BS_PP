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

from typing import Union

import numpy as np
import cv2 as cv


def dbg_show(img):
    """
    Wrapper für Fkt. zum Anzeigen des Bildes;
    zum Debuggen gut.
    :param img: anzuzeigendes Bild
    """
    cv.imshow("DBG", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# # # # # # # # #
# Preprocessing #
# # # # # # # # #


def preprocess(img):
    """
    Wende Weichzeichner an und wandle in Binärbild um.
    :param img: Eingangsbild (RGB!)
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
    """
    Findet die Contour der Hand
    :param mask_img: Binärbild
    :returns: contouren und convexe Hülle des Bildes
    """
    contours, hierarchy = cv.findContours(
        mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    return contours, hull


def get_defects(contours):
    """
    Findet über Convenxe Hülle die Defects der Finger
    :param contours: contourdaten des vorverarbeiteten Bilds
    :returns: defects der Finger
    """
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects


# XXX Punkte liegen nicht genau auf der Tangente!
def find_fingers(img) -> Union[tuple, tuple]:
    """
    Finde die "Täler" zwischen Zeige- und Mittelfinger bzw
    Ring- und kleinem Finger.
    :param img: vorverarbeitetes Bild
    :returns: 2 Koordinaten-Tupel
    """
    mask_img = preprocess(img)
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

    # nur interessante Punkte zurückgeben, rechte Hand 1,3 Linke Hand 0,4
    return finger_points[1], finger_points[3]


def fit(img, p_min, p_max):
    """
    Transformiere Bild so, dass Punkte p_min und p_max
    die y-Achse an der linken Bildkante bilden.
    :param img: Eingangsbild
    :param p_min: Minimum des Koordinatensystems
    :param p_max: Maximum des Koordinatensystems
    :returns: gedrehtes und auf "y-Achse" beschnittenes Bild
    """
    # berechne notwendige Parameter
    a = p_max[0] - p_min[0]
    b = p_min[1] - p_max[1]
    d = round(np.linalg.norm(np.array(p_max) - np.array(p_min)))
    angle = np.rad2deg(np.arctan(a / b))

    # rotiere Bild um p_min
    rot_mat = cv.getRotationMatrix2D(p_min, angle, 1.0)
    rotated = cv.warpAffine(img, rot_mat, img.shape[1::-1])

    # gib (beschnittenes) Bild zurück
    return rotated[p_min[1] - d : p_min[1], p_min[0] : -1]


# ...


def main():
    return 

if __name__ == "__main__":
    sys.exit(main())
