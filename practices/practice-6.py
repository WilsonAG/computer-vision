import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import practices.scripts.operations_over_groups as oog

P_ups = cv.imread("images/ups.jpeg", cv.IMREAD_GRAYSCALE)
N1 = P_ups[35:38, 112:115]
F1 = np.array([
    [1, 2, -3],
    [-2, 3, 4],
    [3, 4, -5]
])

F2 = 1/9 * np.ones((3, 3))

F1_px = oog.myFilter(N1, F1)
img_conv = oog.myConvolution(P_ups, F2)

oog.myCollage(2, 1, P_ups, img_conv)

Fx = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

Fy = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

conv_x = oog.myConvolution(P_ups, Fx)
conv_y = oog.myConvolution(P_ups, Fy)

oog.myCollage(2, 1, conv_x, conv_y)

P_SF = P_ups = cv.imread("images/sanFrancisco.jpg", cv.IMREAD_GRAYSCALE)
edge1 = oog.myEdgeDetector(P_SF, 0.2)
edge2 = oog.myEdgeDetector(P_SF, 0.7)

oog.myCollage(1, 2, edge1, edge2)
