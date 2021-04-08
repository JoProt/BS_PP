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
import more_itertools as mit

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


# # # # # # #
# Constants #
# # # # # # #


THRESH_FACTOR = 0.5

ALPHA = 5
BETA = ALPHA + ALPHA
GAMMA = ALPHA + BETA


# # # # # # # # #
# Preprocessing #
# # # # # # # # #


def check_neighbours(
    pixel: tuple, img: np.ndarray, n: int, r: int, inside: int = 255, outside: int = 0
) -> bool:
    """
    Überprüfe n Nachbarn im Abstand von r, ob sie innerhalb oder außerhalb
    der Fläche im Bild img liegen, ausgehend von Punkt p.
    Anhand dessen kann festgestellt werden, ob es sich um eine Kurve handelt.
    :param p: Koordinatenpunkt
    :param img: Binärbild
    :param n: Anzahl der Nachbarn
    :param r: Abstand der Nachbarn
    :param inside: Farbe, die als "innen" qualifiziert
    :param outside: Farbe, die als "außerhalb der Fläche" gilt
    :returns: Boolean
    """
    stepsize = int(360 / n)
    neighvals = []
    retval = False

    for a in range(0, 360, stepsize):
        d_y = np.round(np.cos(np.deg2rad(a)), 2)
        d_x = np.round(np.sin(np.deg2rad(a)), 2)
        y = int(pixel[0] - (r * d_y))
        x = int(pixel[1] + (r * d_x))

        try:
            nv = img[y, x]
        except IndexError:
            nv = 0

        neighvals.append(nv)

    if 0.5 <= ((sum(neighvals) / 255) / n) <= 0.85:
        retval = True

    return retval


def find_keypoints(img: np.ndarray) -> Union[tuple, tuple]:
    """
    :param img: Eingangsbild (GREY!)
    :returns: ROI / Template zum Matchen
    """
    # weichzeichnen und binarisieren
    blurred = cv.GaussianBlur(img, (13, 13), 0)
    _, thresh = cv.threshold(blurred, (THRESH_FACTOR * img.mean()), 255, cv.THRESH_BINARY)

    # finde die Kontur der Hand
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # mach eine Liste aus Tupeln zur besseren Weiterverarbeitung draus
    contours = [tuple(c[0]) for c in contours[0]]

    # CHVD-Algorithmus, Ong et al.
    valleys = {}

    for i, c in enumerate(contours):
        if (
            check_neighbours(c, img, 4, ALPHA)
            and check_neighbours(c, img, 8, BETA)
            and check_neighbours(c, img, 16, GAMMA)
        ):
            valleys[i] = c
        else:
            pass

    # Sammlung vom Kurven in separate Listen aufteilen; Verbesserungsbedarf!
    keys = [list(group) for group in mit.consecutive_groups(valleys.keys())]
    coords = list(valleys.values())
    valleys = []
    for i in range(len(keys)):
        valleys.append([coords.pop(0) for _ in range(len(keys[i]))])

    return valleys


def crop_to_roi(img: np.ndarray, p_min: tuple, p_max: tuple) -> np.ndarray:
    """
    Transformiere Bild so, dass Punkte p_min und p_max
    die y-Achse an der linken Bildkante bilden.
    :param img: Eingangsbild
    :param p_min: Minimum des Koordinatensystems
    :param p_max: Maximum des Koordinatensystems
    :returns: gedrehtes und auf "y-Achse" beschnittenes Bild
    """
    # berechne notwendige Abstände
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
