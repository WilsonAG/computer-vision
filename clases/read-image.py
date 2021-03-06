from cv2 import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(img)
plt.subplot(3, 2, 2)
plt.imshow(r, cmap='gray')
plt.subplot(3, 2, 3)
plt.imshow(g, cmap='gray')
plt.subplot(3, 2, 4)
plt.imshow(b, cmap='gray')

P = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.imshow(P, cmap='gray')

plt.figure()
tresh, BN = cv2.threshold(P, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.imshow(BN, cmap='gray')

plt.show()

print(img)
