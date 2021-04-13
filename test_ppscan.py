import unittest
import ppscan


class TestPPscan(unittest.TestCase):

    def test_interpol2d(self):
        points = [(0,0),(0,1)]
        steps = 3
        assert ppscan.interpol2d(points, steps) == [(0,0),(0,.5),(0,1)]

    #def test_neighbourhood_curvature(self):
    #    return 0

    #def test_find_valleys(self):
        # drei valleys werden erwartet
    #    return 0

    #def test_find_keypoints(self):
    #    return 0

    #def test_transform_to_roi(self):
    #    return 0