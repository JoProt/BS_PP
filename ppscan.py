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


# # # # # # #
# Constants #
# # # # # # #


THRESH_FACTOR = 0.5

ALPHA = 10
BETA = ALPHA + ALPHA
GAMMA = ALPHA + BETA


# # # # # #
# Utility #
# # # # # #


def interpol2d(points: list, steps: int) -> list:
    """
    Interpoliere steps Punkte auf Basis von points.
    :param points: Liste aus Punkten (Tupel)
    :param steps: Anzahl der Schritte
    :returns: interpolierte Punkte in einer Liste
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    i = np.arange(len(points))
    s = np.linspace(i[0], i[-1], steps)
    return list(zip(np.interp(s, i, x), np.interp(s, i, y)))


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
        # p[]+r nicht innerhalb von img-Dimensionen
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

    return retval


def find_valleys(img: np.ndarray, contour: list) -> list:
    """
    CHVD-Algorithmus, Ong et al. Findet "Täler" in einer Kontur.
    :param img: Bild, in dem die Täler gesucht werden
    :param contour: Kontur des Bildes als Punkteliste
    :returns: Täler als Liste aus Listen von Punkten (nach Zusammenhängen gruppiert)
    """
    valleys = []
    last = 0
    idx = -1

    for i, c in enumerate(contour):
        if (
            neighbourhood_curvature(c, img, 4, ALPHA) == 1.0
            and 0.86 <= neighbourhood_curvature(c, img, 8, BETA) <= 1.0
            # and 0.85 <= neighbourhood_curvature(c, img, 16, GAMMA) <= 1.0
        ):
            if last / i < 0.97:
                idx += 1
                valleys.append([c])
            else:
                valleys[idx].append(c)
            last = i
        else:
            pass

    return valleys


def find_keypoints(img: np.ndarray, hand: int = 0) -> Union[tuple, tuple]:
    """
    Finde die Keypoints des Bildes, in unserem Fall die beiden Lücken
    zwischen Zeige und Mittelfinger bzw. Ring- und kleinem Finger.
    :param img: Eingangsbild
    :param hand: Code der Hand; 0=rechts, 1=links
    :returns: Koordinaten der Keypoints
    """
    # weichzeichnen und binarisieren
    blurred = cv.GaussianBlur(img, (13, 13), 0)
    _, thresh = cv.threshold(blurred, (THRESH_FACTOR * img.mean()), 255, cv.THRESH_BINARY)

    # finde die Kontur der Hand; betrachte nur linke Hälfte des Bildes,
    # weil wir dort die wichtigen Kurven erwarten können
    contours, _ = cv.findContours(
        thresh[:, : int(img.shape[1] / 2)], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # mach eine Liste aus Tupeln zur besseren Weiterverarbeitung draus
    contours = [tuple(c[0]) for c in contours[0]]

    # "Täler" in der Handkontur finden; dort werden sich Keypoints befinden
    valleys = find_valleys(thresh, contours)

    # Werte interpolieren, um eine etwas sauberere Kurve zu bekommen
    valleys_interp = [interpol2d(v, 10) for v in valleys]

    # Anscheinend sind immer der erste und letzte Punkt interessant; stimmt das?
    v_1 = np.array(valleys_interp[0 - hand]).T
    v_2 = np.array(valleys_interp[hand - 1]).T

    # QnD: Mittelpunkt; TODO tatsächlich die Tangente berechnen
    kp_1 = (int(np.round(v_1[0].mean())), int(np.round(v_1[1].mean())))
    kp_2 = (int(np.round(v_2[0].mean())), int(np.round(v_2[1].mean())))

    return kp_1, kp_2


def transform_to_roi(img: np.ndarray, p_min: tuple, p_max: tuple) -> np.ndarray:
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
    # TODO auf Größe des Zuschnitts einigen!
    y_start = p_min[1] - d
    y_end = p_min[1]
    x_start = p_min[0] + 15
    x_end = x_start + 250
    cropped = rotated[y_start:y_end, x_start:x_end]

    return cropped


# ...


def main():
    return


if __name__ == "__main__":
    sys.exit(main())
