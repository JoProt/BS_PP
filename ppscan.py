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

ALPHA = 10
BETA = ALPHA + ALPHA
GAMMA = ALPHA + BETA


# # # # # # # # #
# Preprocessing #
# # # # # # # # #


def neighbourhood_curvature(
    p: tuple, img: np.ndarray, n: int, r: int, inside: int = 255, outside: int = 0
) -> float:
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
    :returns: "Kurvigkeit"
    """
    retval = None
    # Randbehandlung; Kurven in Bildrandgebieten sind nicht relevant!
    if (
        p[0] == 0
        or p[1] == 0
        or p[0] + r >= img.shape[1]
        or p[0] - r < 0
        or p[1] + r >= img.shape[0]
        or p[1] - r < 0
    ):
        retval = 0.0
    else:
        stepsize = int(360 / n)
        neighvals = []

        for a in range(0, 90, stepsize):
            d_y = np.round(np.cos(np.deg2rad(a)), 2)
            d_x = np.round(np.sin(np.deg2rad(a)), 2)
            y_p = int(np.round(p[1] - (r * d_y)))
            y_n = int(np.round(p[1] + (r * d_y)))
            x_p = int(np.round(p[0] + (r * d_x)))
            x_n = int(np.round(p[0] - (r * d_x)))

            neighvals.extend((img[y_p, x_p], img[y_n, x_p], img[y_n, x_n], img[y_p, x_n]))

        retval = (sum(neighvals) / inside) / n
        # print(f"{str(p)} n:{n} r:{r} retval:{retval}")

    return retval


def find_keypoints(img: np.ndarray) -> Union[tuple, tuple]:
    """
    Finde die Keypoints des Bildes, in unserem Fall die beiden Lücken
    zwischen Zeige und Mittelfinger bzw. Ring- und kleinem Finger.
    :param img: Eingangsbild
    :returns: Koordinaten der Keypoints
    """
    # weichzeichnen und binarisieren
    blurred = cv.GaussianBlur(img, (13, 13), 0)
    _, thresh = cv.threshold(blurred, (THRESH_FACTOR * img.mean()), 255, cv.THRESH_BINARY)

    # finde die Kontur der Hand
    # TODO je nach Orientierung beschneiden, da Lücken in bestimmtem Bildausschnitt
    # erwartet werden können.
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # mach eine Liste aus Tupeln zur besseren Weiterverarbeitung draus
    contours = [tuple(c[0]) for c in contours[0]]

    # CHVD-Algorithmus, Ong et al.
    valleys = []
    last = 0
    idx = -1

    for i, c in enumerate(contours):
        if (
            neighbourhood_curvature(c, thresh, 4, ALPHA) == 1.0
            and 0.86 <= neighbourhood_curvature(c, thresh, 8, BETA) <= 1.0
            # and 0.85 <= neighbourhood_curvature(c, thresh, 16, GAMMA) <= 1.0
        ):
            if last / i < 0.97:
                idx += 1
                valleys.append([c])
            else:
                valleys[idx].append(c)
            last = i
        else:
            pass

    # TODO Tangente zwischen 0 und 2 finden; 0 und 2 sind scheinbar immer die relevanten Finger,
    # also sowohl bei linker als auch rechter Hand
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
