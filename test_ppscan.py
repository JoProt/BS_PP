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
        assert ppscan.interpol2d(points, steps) == [
            (0, 0),
            (0, 0.25),
            (0, 0.5),
            (0, 0.75),
            (0, 1),
        ]

    def test_interpol2d_parabel(self):
        """
        Interpolationstest für eine Parabel durch die Punkte (-2,4), (0,0) und (2,4)

        :return: True bei Ausgabe von [(-2.0, 4.0), (-1.0, 2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
        """
        points = [(-2, 4), (0, 0), (2, 4)]
        steps = 5
        assert ppscan.interpol2d(points, steps) == [
            (-2.0, 4.0),
            (-1.0, 2.0),
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 4.0),
        ]

    def test_neighbourhood_curvature_edge(self):
        """
        Testet an der unteren rechten Bild-Ecke (5 Pixel vom Rand entfernt und mit Radius 10 Pixel) ob der erwartete
        Wert von 0.0 ausgegeben wird.

        :return: True wenn neighbourhood_curvature() == 0.0
        """
        img = ppscan.cv.imread("devel/l_01.jpg", ppscan.cv.IMREAD_GRAYSCALE)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature((dim_x - 5, dim_y - 5), img, 10, 10)
        assert val == 0.0

    def test_neighbourhood_curvature_middle(self):
        """
        Testet in der Bildmitte (mit Radius 10 Pixel) ob der ausgegebene Wert >0.0 und <1.0 ist.

        :return: True wenn 0.0 < neighbourhood_curvature() < 1.0
        """
        img = ppscan.cv.imread("devel/l_01.jpg", ppscan.cv.IMREAD_GRAYSCALE)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature(
            (dim_x - (dim_x / 2), dim_y - (dim_y / 2)), img, 10, 10
        )
        assert 0.0 < val < 1.0

    @parameterized.expand(
        [
            ("devel/l_01.jpg",),
            ("devel/l_04.jpg",),
            ("devel/r_03.jpg",),
            ("devel/r_08.jpg",),
        ]
    )
    def test_find_valleys_count(self, file):
        """
        Überprüfung der Listenlänge von valleys (3) und deren Punkte (>=2)

        :return: True, wenn 3 valleys mit je mindestens 2 Punkten gefunden wurden
        """
        # aus find_keypoints() benötigter Code (l.52-59)
        img = ppscan.cv.imread(file, 0)
        blurred = ppscan.cv.GaussianBlur(img, (13, 13), 0)
        _, thresh = ppscan.cv.threshold(
            blurred, (ppscan.THRESH_FACTOR * img.mean()), 255, ppscan.cv.THRESH_BINARY
        )
        contours, _ = ppscan.cv.findContours(
            thresh[:, : int(img.shape[1] / 2)],
            ppscan.cv.RETR_TREE,
            ppscan.cv.CHAIN_APPROX_SIMPLE,
        )
        contours = [tuple(c[0]) for c in contours[0]]
        valleys = ppscan.find_valleys(thresh, contours)
        assert len(valleys) == 3
        for valley in valleys:
            # >= 2, weil interpolation zwischen mindestens zwei Punkten statt finden muss ODER kann es sein,
            # dass nur ein Punkt gefunden werden muss und der dann automatisch als valley gilt?
            assert len(valley) >= 2

    # def test_find_keypoints_l_dummy(self):
    #     """
    #     Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
    #
    #     :return: True bei richtiger Berechnung für Bild l_01.jpg
    #     """
    #     img = ppscan.cv.imread("devel/l_dummy.jpg", ppscan.cv.IMREAD_COLOR)
    #     img = ppscan.cv.cvtColor(img, ppscan.cv.COLOR_BGR2GRAY)
    #     k1, k2 = ppscan.find_keypoints(img)
    #     assert k1 == (169, 188)
    #     assert k2 == (169, 81)
    #
    # def test_find_keypoints_r_dummy(self):
    #     """
    #     Test auf Bestimmung der Keypoints (zwischen Zeige- und Mittelfinger und zwischen Ringfinger und kleinem Finger)
    #
    #     :return: True bei richtiger Berechnung für Bild l_01.jpg
    #     """
    #     img = ppscan.cv.imread("devel/r_dummy.jpg", ppscan.cv.IMREAD_COLOR)
    #     img = ppscan.cv.cvtColor(img, ppscan.cv.COLOR_BGR2GRAY)
    #     k1, k2 = ppscan.find_keypoints(img)
    #     assert k1 == (169, 209)
    #     assert k2 == (169, 319)

    @parameterized.expand(
        [
            ("devel/l_01.jpg",),
            ("devel/l_04.jpg",),
            ("devel/r_03.jpg",),
            ("devel/r_08.jpg",),
        ]
    )
    def test_extract_roi(self, file):
        """
        Testet ob ROI für linke Hand plausibel ist.

        :param file: vollständiges Handbild
        :return: True, wenn ROI dargestellt werden kann
        """
        img_input = ppscan.load_img(file)
        roi = ppscan.extract_roi(img_input)

        assert roi.shape[0] > 0
        assert roi.shape[1] > 0
        assert roi.shape[0] == roi.shape[1]

    def test_build_mask(self):
        """
        Rudimentärer Test der Maskenbildung. Eingabebild zur Generierung der Maske ist vertikal geteilt,
        mit einer Hälfte schwarz, der anderen weiß. Die generierte Maske maskiert alle durch MASK_THRESHOLD
        markierten Bereiche, indem diese in der Maske als schwarze Pixel hinterlegt werden.

        :return: True, wenn Maske und Eingabe Bild übereinstimmen.
        """
        img = ppscan.cv.imread("devel/mask_dummy.jpg", ppscan.cv.IMREAD_GRAYSCALE)
        mask = ppscan.build_mask(img)
        assert mask.all() == img.all()

    def test_papers_GABOR_CONST(self):
        """
        Prüft auf richtige Konstantenwerte (aus dem Paper) für Gabor Filter.

        :return: True, wenn Sigma und Labda Wert denen aus dem Paper entsprechen
        """
        assert ppscan.GABOR_SIGMA == 5.6179
        assert ppscan.GABOR_LAMBDA == 1 / 0.0916

    def test_GABOR_GAMMA_inrange(self):
        """
        Prüft ob Konstante GABOR_GAMMA im richtigen Wertebereich liegt.

        :return: True, wenn Wert von Gamma im Bereich von 0.23 bis 0.92 liegt
        """
        assert 0.23 < ppscan.GABOR_GAMMA < 0.92

    def test_build_gabor_filters_length(self):
        """
        Weis eigentlich nicht so recht, was man für die Funktion testen sollte... Außer ob die ausgegebene Liste
        wirklich genau so lang ist wie die Liste von vorgegebenen Thetas.

        :return: True, wenn Filterliste genau so lang wie Liste von Thetas
        """
        filters = ppscan.build_gabor_filters()
        assert ppscan.GABOR_THETAS.size == len(filters)

    def test_apply_gabor_filters_imgsize(self):
        """
        Testet auf gleiche Dimensionen von Filter-Eingangsbild und Filter-Ausgabebild.

        :return: True bei gleichen Dimensionen von zu filterndem und gefiltertem Bild
        """
        img = ppscan.cv.imread("devel/r_dummy.jpg", ppscan.cv.IMREAD_GRAYSCALE)
        filters = ppscan.build_gabor_filters()
        merged_img = ppscan.apply_gabor_filters(img, filters)
        assert merged_img.shape == img.shape

    def test_find_tangent_points_NONE(self):
        v_1 = []
        v_2 = []
        test1, test2 = ppscan.find_tangent_points(v_1, v_2)
        assert test1 == None
        assert test2 == None

    def test_find_keypoints_exceptions(self):
        valleys = []
        with self.assertRaises(Exception) as context:
            ppscan.find_keypoints(valleys)
        self.assertTrue("Expected at least 2 valleys!" in str(context.exception))

        valleys = [[[0, 0]],[[0, 0]]]
        with self.assertRaises(Exception) as context:
            ppscan.find_keypoints(valleys)
        self.assertTrue(
            "No valleys found!" in str(context.exception)
        )

    def test_db_access(self):
        """
        Testet die Verbindung zur Datenbank.

        :return: True, wenn peter die Nummer 1 ist
        """
        peter = ppscan.session.query(ppscan.User).first()
        assert peter.id == 1
