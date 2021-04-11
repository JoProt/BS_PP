import ppscan

img = ppscan.cv.imread("devel/r_03.jpg", 0)
# ppscan.dbg_show(img)
k1, k2 = ppscan.find_keypoints(img)

roi = ppscan.transform_to_roi(img, k2, k1)

ppscan.dbg_show(roi)
