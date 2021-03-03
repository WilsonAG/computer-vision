import cv2
import numpy as np
import matplotlib.pyplot as plt

p1 = cv2.imread("pr6-images/ups.jpeg", )
p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2RGB)
P_ups = cv2.cvtColor(p1, cv2.COLOR_RGB2GRAY)

N1 = P_ups[35:38, 112:115]


def myFilter(N, F):
    F2 = np.flip(F, 0)
    F3 = np.flip(F2, 1)
    multiplicacion = np.multiply(N, F3)
    suma = np.sum(multiplicacion)
    return suma


F1 = np.array([
    [1, 2, -3],
    [-2, 3, 4],
    [3, 4, -5]
])

mtx = myFilter(N1, F1)


def myConvolution(P, F):
    x = np.pad(P, pad_width=1, mode='constant', constant_values=0)
    n_filas = np.size(P, 0)
    # for i in range(n_filas)

    return x