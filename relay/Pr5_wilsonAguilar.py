# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:32:55 2020

@author: User
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

#### Literal 1
p1 = cv2.imread("pr5-images/foto1.jpg")  # leyendo la imagen
p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2RGB)  # rgb
p1 = cv2.cvtColor(p1, cv2.COLOR_RGB2GRAY)

p2 = cv2.imread("pr5-images/foto2.jpg")  # leyendo la imagen
p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)  # rgb
p2 = cv2.cvtColor(p2, cv2.COLOR_RGB2GRAY)

p3 = cv2.imread("pr5-images/foto3.jpg")  # leyendo la imagen
p3 = cv2.cvtColor(p3, cv2.COLOR_BGR2RGB)  # rgb
p3 = cv2.cvtColor(p3, cv2.COLOR_RGB2GRAY)

p4 = cv2.imread("pr5-images/foto4.jpg")  # leyendo la imagen
p4 = cv2.cvtColor(p4, cv2.COLOR_BGR2RGB)  # rgb
p4 = cv2.cvtColor(p4, cv2.COLOR_RGB2GRAY)


# cuantos valores se repiten

def miHistograma(P):
    pixeles = []
    for k in range(256):
        a = np.where(P == k)
        pixeles.append(np.size(a[0]))
    return np.array(pixeles)


def miHPA(P):
    t = P.shape[0] * P.shape[1]
    a = miHistograma(P)
    salida = np.cumsum(a)
    return salida / t


p1_hpa = miHPA(p1)
p1_hist = miHistograma(p1)
p2_hpa = miHPA(p2)
p2_hist = miHistograma(p2)
p3_hpa = miHPA(p3)
p3_hist = miHistograma(p3)
p4_hpa = miHPA(p4)
p4_hist = miHistograma(p4)

plt.figure()

plt.subplot(3, 4, 1)
plt.imshow(p1, cmap='gray')
plt.subplot(3, 4, 2)
plt.imshow(p2, cmap='gray')
plt.subplot(3, 4, 3)
plt.imshow(p3, cmap='gray')
plt.subplot(3, 4, 4)
plt.imshow(p4, cmap='gray')

plt.subplot(3, 4, 5)
plt.plot(p1_hist)
plt.subplot(3, 4, 6)
plt.plot(p2_hist)
plt.subplot(3, 4, 7)
plt.plot(p3_hist)
plt.subplot(3, 4, 8)
plt.plot(p4_hist)

plt.subplot(3, 4, 9)
plt.plot(p1_hpa)
plt.subplot(3, 4, 10)
plt.plot(p2_hpa)
plt.subplot(3, 4, 11)
plt.plot(p3_hpa)
plt.subplot(3, 4, 12)
plt.plot(p4_hpa)

plt.show()


