import unittest
import ppscan
from parameterized import parameterized


class TestPPscan(unittest.TestCase):

    def test_interpol2d_gerade(self):
        """
        Interpolationstest für eine Gerade durch die Punkte (0,0) und (0,1)
        :return: True bei Ausgabe von [(0, 0), (0, .25), (0, .5), (0, .75), (0, 1)]
        """
        points = [(0, 0), (0, 1)]
        steps = 5
        assert ppscan.interpol2d(points, steps) == [(0, 0), (0, .25), (0, .5), (0, .75), (0, 1)]

    def test_interpol2d_parabel(self):
        """
        Interpolationstest für eine Parabel durch die Punkte (-2,4), (0,0) und (2,4)
        :return: True bei Ausgabe von [(-2.0, 4.0), (-1.0, 2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
        """
        points = [(-2, 4), (0, 0), (2, 4)]
        steps = 5
        assert ppscan.interpol2d(points, steps) == [(-2.0, 4.0), (-1.0, 2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]

    def test_neighbourhood_curvature_edge(self):
        """
        Testet an der unteren rechten Bild-Ecke (5 Pixel vom Rand entfernt und mit Radius 10 Pixel) ob der erwartete Wert von 0.0 ausgegeben wird.
        :return: True wenn neighbourhood_curvature() == 0.0
        """
        img = ppscan.cv.imread("devel/l_01.jpg", 0)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature((dim_x - 5, dim_y - 5), img, 10, 10)
        assert val == 0.0

    def test_neighbourhood_curvature_middle(self):
        """
        Testet in der Bildmitte (mit Radius 10 Pixel) ob der ausgegebene Wert >0.0 und <1.0 ist.
        :return: True wenn 0.0 < neighbourhood_curvature() < 1.0
        """
        img = ppscan.cv.imread("devel/l_01.jpg", 0)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature((dim_x - (dim_x / 2), dim_y - (dim_y / 2)), img, 10, 10)
        assert 0.0 < val < 1.0

    # def test_find_valleys(self):
    # drei valleys werden erwartet
    #    return 0

    def test_find_keypoints_l_01(self):
        """
        Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
        :return: True bei richtiger Berechnung für Bild l_01.jpg
        """
        img = ppscan.cv.imread("devel/l_01.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (221, 155)
        assert k2 == (142, 309)

    def test_find_keypoints_l_04(self):
        """
        Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
        :return: True bei richtiger Berechnung für Bild l_04.jpg
        """
        img = ppscan.cv.imread("devel/l_04.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (175, 119)
        assert k2 == (95, 276)

    def test_find_keypoints_r_03(self):
        """
        Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
        :return: True bei richtiger Berechnung für Bild r_03.jpg
        """
        img = ppscan.cv.imread("devel/r_03.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (117, 179)
        assert k2 == (171, 356)

    def test_find_keypoints_r_08(self):
        """
        Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
        :return: True bei richtiger Berechnung für Bild r_08.jpg
        """
        img = ppscan.cv.imread("devel/r_08.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (164, 201)
        assert k2 == (198, 379)

    # def test_transform_to_roi(self):
    #    return 0
