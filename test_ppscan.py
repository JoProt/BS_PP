import unittest
import ppscan
from parameterized import parameterized


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

    def test_find_keypoints_l_01(self):
        img = ppscan.cv.imread("devel/l_01.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (221, 155)
        assert k2 == (142, 309)

    def test_find_keypoints_l_04(self):
        img = ppscan.cv.imread("devel/l_04.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (175, 119)
        assert k2 == (95, 276)

    def test_find_keypoints_r_03(self):
        img = ppscan.cv.imread("devel/r_03.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (117, 179)
        assert k2 == (171, 356)

    def test_find_keypoints_r_08(self):
        img = ppscan.cv.imread("devel/r_08.jpg", 0)
        k1, k2 = ppscan.find_keypoints(img)
        assert k1 == (164, 201)
        assert k2 == (198, 379)

    #def test_transform_to_roi(self):
    #    return 0