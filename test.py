import ppscan
import numpy as np
import more_itertools as mit
import itertools

img = ppscan.cv.imread(
    "/home/lmg/HSW/BiomSys/Praktikum/CASIA-PalmprintV1.zi__FILES/0001/0001_m_l_07.jpg", 0
)

v = ppscan.find_keypoints(img)
# v_ = [list(group) for k, g in mit.consecutive_groups(v.keys())]

print(v)

for a in v:
    for p in a:
        ppscan.cv.circle(img, p, 1, (255, 255, 255), -1)

ppscan.dbg_show(img)

# drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# for a in v:
#     pts = ppscan.np.array([list(x) for x in a])
#     pts.reshape((-1, 1, 2))
#     ppscan.cv.polylines(drawing, [pts], False, (255, 255, 255), 1)

# ppscan.dbg_show(drawing)
