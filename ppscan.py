#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Palmprint Scanner
    =================
    [your ad here]
    
    :license: who knows
    :format: black, reST docstrings
"""

import sys

from typing import Union

import numpy as np
import cv2 as cv

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

from scipy.spatial import distance

# Verbindung zur Datenbank
engine = create_engine("sqlite:///palmprint.db")

# Datenmodell laden
Base = declarative_base()

# Session für Datenbankzugriff erzeugen
Session = sessionmaker(bind=engine)
session = Session()


# # # # # # #
# Constants #
# # # # # # #


THRESH_FACTOR = 0.5
THRESH_SUBIMG = 150.0
THRESH_CON = 15

GAMMA = 13
G_L = 24 / 32
G_U = 30 / 32

GABOR_KSIZE = (35, 35)
GABOR_SIGMA = 5.6179
GABOR_THETAS = np.arange(0, np.pi, np.pi / 32)
GABOR_LAMBDA = 1 / 0.0916
GABOR_GAMMA = 0.7  # 0.23 < gamma < 0.92
GABOR_PSI = 0
GABOR_THRESHOLD = 255  # 0 to 255

# XXX das ist etwas hoch ... r_08 z.B. wird maskiert, was nicht sein muss
MASK_THRESHOLD = 110


# # # # # #
# Models  #
# # # # # #


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    palmprints = relationship("Palmprint")

    def __repr__(self):
        return "<{} {} '{}', {} prints registered>".format(
            self.__class__.__name__, self.id, self.name, len(self.palmprints)
        )


class Palmprint(Base):
    __tablename__ = "palmprints"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    data = Column(String)

    def __repr__(self):
        return "<{} {} (user {})>".format(
            self.__class__.__name__, self.id, self.user_id
        )


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


def find_tangent_points(v_1: list, v_2: list) -> Union[tuple, tuple]:
    """
    Prüfe für jeden Punkt P in einem Tal (Kurve K), ob eine Gerade zwischen P
    und einem Punkt auf der anderen Kurve eine Tangente beider Kurven ist.
    Wenn ja, gib jeweils die Punkte beider Kurven zurück, die auf der Tangente liegen.

    :param v_1, v_2: zu betrachtende Kurven (Listen aus Koordinatentupeln)
    :returns: Punkte der Tangente bei Existenz, None andernfalls
    """
    vs = v_1 + v_2

    # wenn die Gerade zwischen p_1 und p_2 keine weiteren Punkte in v_1 und v_2 schneidet,
    # dann ist die Gerade als Tangente beider Kurven anzusehen
    for p_1 in v_1:
        for p_2 in v_2:
            # Wahrheitskriterium soll auf alle p aus v_1 und v_2 zutreffen
            if all(
                [
                    # ist f(p.y) größer als p.x, d.h. existiert kein Schnittpunkt?
                    # f sei die Geradengleichung der Geraden zw. p_1 und p_2
                    (
                        p_1[0] * ((p[1] - p_2[1]) / (p_1[1] - p_2[1]))
                        + p_2[0] * ((p[1] - p_1[1]) / (p_2[1] - p_1[1]))
                    )
                    >= p[0]
                    for p in vs
                ]
            ):
                # runde Koordinaten zu nächsten ganzzahligen Pixelwerten
                p_1 = (int(np.round(p_1[0])), int(np.round(p_1[1])))
                p_2 = (int(np.round(p_2[0])), int(np.round(p_2[1])))
                return p_1, p_2

    return None, None


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

            neighvals.extend(
                (img[y_p, x_p], img[y_n, x_p], img[y_n, x_n], img[y_p, x_n])
            )

        retval = (sum(neighvals) / inside) / n

    return retval


def find_valleys(img: np.ndarray, contour: list) -> list:
    """
    Kombiniert den CHVD-Algorithmus (Ong et al.) mit einfachem Mask-Matching
    und findet so die "Täler" in einer Kontur.

    :param img: Bild, in dem die Täler gesucht werden
    :param contour: Kontur des Bildes als Punkteliste
    :returns: Täler als Liste aus Listen von Punkten (nach Zusammenhängen gruppiert)
    """
    valleys = []
    last = 0
    idx = -1

    # durchlaufe die Punkte auf der Kontur
    for i, c in enumerate(contour):
        # quadratischer Bildausschnitt der Seitenlänge GAMMA mit c als Mittelpunkt
        subimg = img[c[1] - GAMMA : c[1] + GAMMA, c[0] - GAMMA : c[0] + GAMMA]
        if (
            len(subimg) > 0
            and (G_L <= neighbourhood_curvature(c, img, 32, GAMMA) <= G_U)
            and subimg.mean() >= THRESH_SUBIMG
        ):
            # prüfe auf mögl. Zusammenhang mit vorheriger Gruppe; starte neue Gruppe,
            # wenn der Abstand zu groß ist
            if i - last > THRESH_CON:
                idx += 1
                valleys.append([c])
            else:
                valleys[idx].append(c)
            last = i

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
    blurred = cv.GaussianBlur(img, (7, 7), 0)
    _, thresh = cv.threshold(
        blurred, (THRESH_FACTOR * img.mean()), 255, cv.THRESH_BINARY
    )

    # finde die Kontur der Hand; betrachte nur linke Hälfte des Bildes,
    # weil wir dort die wichtigen Kurven erwarten können
    contours, _ = cv.findContours(
        thresh[:, : int(img.shape[1] / 2)], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # filtere nur die längste Kontur, um mögl. Störungen zu entfernen
    contour = max(contours, key=lambda c: len(c))

    # mach eine Liste aus Tupeln zur besseren Weiterverarbeitung draus
    contour = [tuple(c[0]) for c in contour]

    # "Täler" in der Handkontur finden; dort werden sich Keypoints befinden
    valleys = find_valleys(thresh, contour)

    if len(valleys) < 2:
        raise Exception("Expected at least 2 valleys!")

    # schmeiße 1er-Längen raus, sind meistens Fehler
    valleys = [v for v in valleys if len(v) > 1]

    # sortiere die Täler nach ihrer y-Koordinate
    valleys.sort(key=lambda v: v[0][1])

    # Werte interpolieren, um eine etwas sauberere Kurve zu bekommen
    valleys_interp = [interpol2d(v, 10) for v in valleys]

    # im Optimalfall sollten erster und letzter Punkt die Keypoints sein
    v_1 = valleys_interp[0 - hand]
    v_2 = valleys_interp[hand - 1]

    # Punkte auf Tangente beider Täler finden; das sind die Keypoints
    kp_1, kp_2 = find_tangent_points(v_1, v_2)

    if kp_1 is None or kp_2 is None:
        raise Exception("Couldn't find a tangent for {} and {}!".format(kp_1, kp_2))

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
    y_start = p_min[1] - d
    y_end = p_min[1] + 70
    x_start = p_min[0] + 10
    x_end = x_start + 350
    cropped = rotated[y_start:y_end, x_start:x_end]

    return cropped


def build_mask(img: np.ndarray) -> np.ndarray:
    """
    Generiert eine Maske aus dem gegebenen Bild.

    :param img: Bild welches als Grundlage der Maske dienen soll
    :return: Maske (schwarz/weiß)
    """
    # generiere leeres np array, füllen mit 'weiß' (255)
    mask = np.empty_like(img)
    mask.fill(255)
    # setze jedes Maskenpixel auf 0, wenn im gegebenen Bildpixel der Wert kleiner als
    # der Schwellwert ist
    mask[img < MASK_THRESHOLD] = 0

    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Gegebene Maske auf gegebenes Bild anwenden.

    :param img: zu maskierendes Bild
    :param mask: Maske des Bildes
    :return: maskiertes Bild
    """
    white_background = np.empty_like(img)
    white_background.fill(255)

    inv_mask = cv.bitwise_not(mask)
    # cv.imshow("inv_mask", inv_mask)
    img1 = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow("img1", img1)
    img2 = cv.bitwise_and(white_background, white_background, mask=inv_mask)
    # cv.imshow("img2", img2)
    masked_img = cv.add(img1, img2)
    # masked_img = cv.bitwise_and(img, img, mask=mask)

    return masked_img


def build_gabor_filters() -> list:
    """
    Generiert eine Liste von Gabor Filtern aus gegebenen Konstanten.

    :return: Liste von Gabor Filtern
    """
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    filters = []

    for theta in GABOR_THETAS:
        params = {
            "ksize": GABOR_KSIZE,
            "sigma": GABOR_SIGMA,
            "theta": theta,
            "lambd": GABOR_LAMBDA,
            "gamma": GABOR_GAMMA,
            "psi": GABOR_PSI,
            "ktype": cv.CV_32F,
        }
        kern = cv.getGaborKernel(**params)
        filters.append((kern, params))

    return filters


def apply_gabor_filters(img: np.ndarray, filters: list) -> np.ndarray:
    """
    Wendet Filter-Liste auf Bildkopien an, fügt gefilterte Bilder zu einem zusammen
    und entfernt alle Elemente unter einem festgelegten Schwellwert.

    :param img: zu filterndes Bild
    :param filters: Liste von Gabor Filtern
    :return: gefiltertes Bild
    """
    # generate empty np array and fill it with 'white' (255)
    merged_img = np.empty_like(img)
    merged_img.fill(255)

    # for all filters: filter image and merge with already filtered images
    for kern, params in filters:
        filtered_img = cv.filter2D(img, cv.CV_8UC3, kern)
        np.minimum(merged_img, filtered_img, merged_img)

    # use threshold to remove lines
    merged_img[merged_img > GABOR_THRESHOLD] = 255

    return merged_img


# ...


def match_Palm_Prints(img_to_match: np.ndarray, img_template: np.ndarray) -> bool:
    """
    Vergleicht aktuelles Image mit Images aus Datenbank und sucht Match.
    Rueckgabewert: 1 -> Images matchen
    Rueckgabewert: 0 -> Images matchen nicht

    :param img_to_match: abzugleichendes Image
    :param template_image: Vorlage, gegen welche gematched wird
    :return: gibt Match (1) oder Non Match (0) zurueck
    """

    matching_decision: bool = 0

    hamming_distance = distance.hamming([1, 2, 3], [1, 2, 4])

    print("hamming distance: ", hamming_distance)
    if hamming_distance <= 0.2:
        matching_decision = 1

    print("matching_decision: ", matching_decision)

    return matching_decision


def main():
    img = cv.imread("devel/r_03.jpg", cv.IMREAD_GRAYSCALE)
    k1, k2 = find_keypoints(img)
    roi = transform_to_roi(img, k2, k1)
    # cv.imshow("roi", roi)

    mask = build_mask(roi)
    # cv.imshow("mask", mask)

    filters = build_gabor_filters()
    filtered_roi = apply_gabor_filters(roi, filters)
    # cv.imshow("filtered_roi", filtered_roi)

    masked_roi = apply_mask(filtered_roi, mask)

    cv.imshow("masked_roi", masked_roi)

    # --Creating 2nd Image for Testing purpose----------------------------------------------------------------------------

    img_template = cv.imread("devel/r_03.jpg", cv.IMREAD_GRAYSCALE)
    k1_template, k2_template = find_keypoints(img_template)
    roi_template = transform_to_roi(img_template, k2_template, k1_template)
    # cv.imshow("roi", roi)

    mask_template = build_mask(roi_template)
    # cv.imshow("mask", mask)

    filters_template = build_gabor_filters()
    filtered_roi_template = apply_gabor_filters(roi_template, filters_template)
    # cv.imshow("filtered_roi", filtered_roi)

    masked_roi_template = apply_mask(filtered_roi, mask_template)

    cv.imshow("masked_roi_template", masked_roi_template)

    # -------------------------------------------------------------------------------------

    match_Palm_Prints(masked_roi, masked_roi_template)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
