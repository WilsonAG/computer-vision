import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import pickle

with open('./data.pkl', 'rb') as f:
    data = pickle.load(f)

s1 = data['s1']
s2 = data['s2']
s3 = data['s3']
s4 = data['s4']

plt.figure()
plt.subplot(221)
plt.title("S1")
plt.imshow(s1, cmap='gray')

plt.subplot(222)
plt.title("S2")
plt.imshow(s2, cmap='gray')

plt.subplot(223)
plt.title("S3")
plt.imshow(s3, cmap='gray')

plt.subplot(224)
plt.title("S4")
plt.imshow(s4, cmap='gray')


plt.show()


def miMomento(P, p, q):
    sumatoria = 0
    n_filas = np.size(P, 0)
    n_columnas = np.size(P, 1)

    for i in range(n_filas):
        for j in range(n_columnas):
            sumatoria = sumatoria + ((i + 1) ** p * (j + 1) ** q * P[i, j])
    return sumatoria


M01 = miMomento(s3, 0, 1)


def miMomentoCentral(P, p, q):
    x_techo = miMomento(P, 1, 0) / miMomento(P, 0, 0)
    y_techo = miMomento(P, 0, 1) / miMomento(P, 0, 0)
    sumatoria = 0
    n_filas = np.size(P, 0)
    n_columnas = np.size(P, 1)

    for i in range(n_filas):
        for j in range(n_columnas):
            sumatoria = sumatoria + ((i + 1 - x_techo) ** p * (j + 1 - y_techo) ** q * P[i, j])
    return sumatoria


_u12 = miMomentoCentral(s3, 1, 2)


def misMomentosHu(P):
    u20 = miMomentoCentral(P, 2, 0)
    u02 = miMomentoCentral(P, 0, 2)
    u11 = miMomentoCentral(P, 1, 1)
    u30 = miMomentoCentral(P, 3, 0)
    u03 = miMomentoCentral(P, 0, 3)
    u12 = miMomentoCentral(P, 1, 2)
    u21 = miMomentoCentral(P, 2, 1)

    h1 = u20 + u02
    h2 = (u20 - u02) ** 2 + 4 * u11 ** 2
    h3 = (u30 - 3 * u12) ** 2 + (3 * u21 - u03) ** 2
    h4 = (u30 + u12) ** 2 + (u21 + u03) ** 2

    return np.array([h1, h2, h3, h4])


def misMomentosHu2(P):
    u20 = miMomentoCentral(P, 2, 0)
    u02 = miMomentoCentral(P, 0, 2)
    u11 = miMomentoCentral(P, 1, 1)

    h1 = u20 + u02
    h2 = (u20 - u02) ** 2 + 4 * u11 ** 2
    return np.array([h1, math.sqrt(h2)])


mh_s1 = misMomentosHu(s1)
mh_s2 = misMomentosHu(s2)
mh_s3 = misMomentosHu(s3)
mh_s4 = misMomentosHu(s4)

letters = []
scale_percent = 25

for i in range(97, 123):
    img = cv.imread('./pr8/letters/' + chr(i) + '.jpg', cv.IMREAD_GRAYSCALE)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    letters.append(img)

text = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


fig, ax = plt.subplots()
x_points = []
y_points = []

for i in range(len(letters)):
    x, y = misMomentosHu2(letters[i])
    x_points.append(x)
    y_points.append(y)

ax.scatter(x_points, y_points)

for i, txt in enumerate(text):
    ax.annotate(txt, (x_points[i], y_points[i]))

plt.show()

