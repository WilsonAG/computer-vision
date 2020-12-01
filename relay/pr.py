import cv2
import matplotlib.pyplot as plt

img = cv2.imread("automovil.jpg")
#
p = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# new_img = 255 - p
#
# r = new_img[:, :, 0]
# g = new_img[:, :, 1]
# b = new_img[:, :, 2]
#
# # new_img[:,:,0] 325 275
# new_img[325:330, 275:280, 1] = 255
#
# plt.imshow(new_img)
# plt.show()


def miRGB2gray(P, alfa, beta, gamma):
    r = P[:, :, 0]
    g = P[:, :, 1]
    b = P[:, :, 2]

    Q = alfa * r + beta * g + b * gamma
    return Q


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(p)

q = miRGB2gray(p, 1 / 3, 1 / 3, 1 / 3)
plt.subplot(1, 2, 2)
plt.imshow(q, cmap='gray')
plt.show()

im1 = miRGB2gray(p, .1, .6, .3)
im2 = miRGB2gray(p, .6, .3, .1)
im3 = miRGB2gray(p, .3, .1, .6)

plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(im1, cmap='gray')
plt.subplot(3, 1, 2)
plt.imshow(im2, cmap='gray')
plt.subplot(3, 1, 3)
plt.imshow(im3, cmap='gray')

plt.show()
