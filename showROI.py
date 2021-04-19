import cv2 as cv
import ppscan
import matplotlib.pyplot as plt
img = cv.imread("devel/r_03.jpg", 0)
k1, k2 = ppscan.find_keypoints(img)
roi = ppscan.transform_to_roi(img, k2, k1)
ppscan.dbg_show(roi)