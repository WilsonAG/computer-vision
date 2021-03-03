import numpy as np
import matplotlib.pyplot as plt


def myFilter(N, F):
    F = np.flip(F)
    px = np.multiply(F, N).sum()
    return px


def myConvolution(P, F):
    X = np.empty(P.shape)
    P_padd = np.pad(P, 1)
    rows, cols = P.shape
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            N = P_padd[i - 1: i + 2, j - 1:j + 2]
            X[i - 1, j - 1] = myFilter(N, F)
    return X


def myCollage(nrows, ncols, im1, im2):
    plt.figure()
    plt.subplot(nrows, ncols, 1)
    plt.imshow(im1, cmap='gray')
    plt.subplot(nrows, ncols, 2)
    plt.imshow(im2, cmap='gray')
    plt.show()


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


def myEdgeDetector(P, thresh):
    Px = myConvolution(P, Fx)
    Py = myConvolution(P, Fy)
    X = Px + Py
    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            X[i, j] = 255 if X[i, j] > thresh * 255 else 0
    return X

