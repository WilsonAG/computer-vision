import cv2
import matplotlib.pyplot as plt
import numpy as np
from practices.scripts import color_spaces as cs

# # Literal 1
#
# img = cv2.imread("images/face.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
#
# inverse_img = cs.invert_image(img)
#
# plt.subplot(1, 2, 2)
# plt.imshow(inverse_img, cmap='gray')
# plt.show()
#
# # Literal 2
# img = cv2.imread("images/automovil.jpg")
# img_grayscale = cs.miRGB2gray(img, 1 / 3, 1 / 3, 1 / 3)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img_grayscale, cmap='gray')
# plt.show()
#
# # Literal 3
#
# img_grayscale1 = cs.miRGB2gray(img, .1, .6, .3)
# img_grayscale2 = cs.miRGB2gray(img, .6, .3, .1)
# img_grayscale3 = cs.miRGB2gray(img, .3, .1, .6)
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(img_grayscale1, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(img_grayscale2, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(img_grayscale3, cmap='gray')
# plt.show()
#
# # Literal 4
# img = cv2.imread("images/automovil.jpg")
# img_grayscale = cs.miRGB2gray(img, 1 / 3, 1 / 3, 1 / 3)
#
# plt.figure()
# plt.hist(img_grayscale.ravel(), 256, [0, 256])
# plt.show()
#
# # Literal 5
# img1 = cv2.imread("images/imagen iluminada.jpg")
# img2 = cv2.imread("images/imagen oscura.jpg")
#
# img1_grayscale = cs.miRGB2gray(img1, 1 / 3, 1 / 3, 1 / 3)
# img2_grayscale = cs.miRGB2gray(img2, 1 / 3, 1 / 3, 1 / 3)
#
# h1 = cs.miHistograma(img1_grayscale)
# h2 = cs.miHistograma(img2_grayscale)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.hist(img1_grayscale.ravel(), 256, [0, 256])
#
# plt.subplot(1, 2, 2)
# plt.hist(img2_grayscale.ravel(), 256, [0, 256])
#
# plt.show()


# # Literal 6
# P = np.array([
#     [255, 170, 55, 20],
#     [5, 34, 165, 240],
#     [120, 80, 64, 243],
#     [95, 99, 23, 114]
# ])
#
# binary = cs.miGray2binaria(P, 120)

# Literal 7
img = cv2.imread("images/automovil.jpg", cv2.COLOR_BGR2RGB)
img_gray = cs.miRGB2gray(img, 1 / 3, 1 / 3, 1 / 3)
binary1 = cs.miGray2binaria(img_gray, 10)
binary2 = cs.miGray2binaria(img_gray, 245)

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(binary1, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(binary2, cmap='gray')
# plt.show()

# Literal 8
phi = cs.miOtsu(img_gray)

# Literal 9
bn_img = cs.miGray2binaria(img_gray, 108)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(bn_img, cmap='gray')
plt.show()
