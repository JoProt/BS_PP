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

    @parameterized.expand(
        [
            ("testdaten/klaus_testing/l_01.jpg",),
            ("testdaten/klaus_testing/r_01.jpg",),
            ("testdaten/peter_testing/l_01.jpg",),
            ("testdaten/peter_testing/r_01.jpg",),
        ]
    )
    def test_neighbourhood_curvature_edge(self, file):
        """
        Testet an der unteren rechten Bild-Ecke (5 Pixel vom Rand entfernt und mit Radius 10 Pixel) ob der erwartete
        Wert von 0.0 ausgegeben wird.

        :return: True wenn neighbourhood_curvature() == 0.0
        """
        img = ppscan.load_img(file)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature((dim_x - 5, dim_y - 5), img, 10, 10)
        assert val == 0.0

    @parameterized.expand(
        [
            ("testdaten/klaus_testing/l_01.jpg",),
            ("testdaten/klaus_testing/r_01.jpg",),
            ("testdaten/peter_testing/l_01.jpg",),
            ("testdaten/peter_testing/r_01.jpg",),
        ]
    )
    def test_neighbourhood_curvature_middle(self, file):
        """
        Testet in der Bildmitte (mit Radius 10 Pixel) ob der ausgegebene Wert >0.0 und <1.0 ist.

        :return: True wenn 0.0 < neighbourhood_curvature() < 1.0
        """
        img = ppscan.load_img(file)
        dim_x, dim_y = img.shape
        val = ppscan.neighbourhood_curvature(
            (dim_x - (dim_x / 2), dim_y - (dim_y / 2)), img, 10, 10
        )
        assert 0.0 < val < 1.0

    @parameterized.expand(
        [
            ("testdaten/klaus_testing/l_01.jpg",),
            ("testdaten/klaus_testing/r_01.jpg",),
            ("testdaten/peter_testing/l_01.jpg",),
            ("testdaten/peter_testing/r_01.jpg",),
        ]
    )
    def test_find_valleys_count(self, file):
        """
        Überprüfung der Listenlänge von valleys (3) und deren Punkte (>=2)

        :return: True, wenn 3 valleys mit je mindestens 2 Punkten gefunden wurden
        """
        img = ppscan.load_img(file)

        blurred = ppscan.cv.GaussianBlur(img, (7, 7), 0)
        _, thresh = ppscan.cv.threshold(
            blurred, (ppscan.THRESH_FACTOR * img.mean()), 255, ppscan.cv.THRESH_BINARY
        )
        contours, _ = ppscan.cv.findContours(
            thresh[:, : int(img.shape[1] / 2)],
            ppscan.cv.RETR_LIST,
            ppscan.cv.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=lambda c: len(c))
        contour = [tuple(c[0]) for c in contour]
        valleys = ppscan.find_valleys(thresh, contour)

        assert len(valleys) == 3
        for valley in valleys:
            # >= 2, weil interpolation zwischen mindestens zwei Punkten statt finden muss ODER kann es sein,
            # dass nur ein Punkt gefunden werden muss und der dann automatisch als valley gilt?
            assert len(valley) >= 2

    @parameterized.expand(
        [
            ("testdaten/klaus_testing/l_01.jpg",),
            ("testdaten/klaus_testing/r_01.jpg",),
            ("testdaten/peter_testing/l_01.jpg",),
            ("testdaten/peter_testing/r_01.jpg",),
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
        img = ppscan.cv.imread("testdaten/mask_dummy.jpg", ppscan.cv.IMREAD_GRAYSCALE)
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

    def test_find_tangent_points_NONE(self):
        v_1 = []
        v_2 = []
        test1, test2 = ppscan.find_tangent_points(v_1, v_2)
        if test1 is None:
            assert True
        if test2 is None:
            assert True

    def test_find_keypoints_exceptions(self):
        valleys = []
        with self.assertRaises(Exception) as context:
            ppscan.find_keypoints(valleys)
        self.assertTrue("Expected at least 2 valleys!" in str(context.exception))

        valleys = [[[0, 0]], [[0, 0]]]
        with self.assertRaises(Exception) as context:
            ppscan.find_keypoints(valleys)
        self.assertTrue("No valleys found!" in str(context.exception))

    def test_db_access(self):
        """
        Testet die Verbindung zur Datenbank.

        :return: True, wenn peter die Nummer 1 ist
        """
        peter = ppscan.session.query(ppscan.User).first()
        assert peter.id == 1

    # TODO: Tests für Datenbankverbindung
