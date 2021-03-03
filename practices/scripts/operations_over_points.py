import numpy as np
import math


def miHPA(P):
    val, counter = np.unique(P, return_counts=True)
    h = np.zeros(256)

    for idx, count in zip(val, counter):
        h[idx] = count

    pxls = P.size
    k = h.cumsum() / pxls
    return k


def equalize(P):
    Q = np.empty(P.shape)
    h = np.zeros(256)
    c = np.zeros(256)
    i, j = P.shape
    freq = np.unique(P, return_counts=True)
    # Hystogram
    for (idx, val) in zip(freq[0], freq[1]):
        h[idx] = val
    # HPA
    for idx in range(len(h)):
        c[idx] = (1 / (i * j)) * sum(h[0:idx + 1])
    # print(c[255])
    # apply HPA to mtx
    for a in range(i):
        for b in range(j):
            Q[a, b] = c[P[a, b]]

    return (Q * 255).round()


def normalize(P):
    Q = np.empty(P.shape)
    m = 255 / (P.max() - P.min())
    i, j = P.shape
    print(m)
    for a in range(i):
        for b in range(j):
            Q[a, b] = m * (P[a, b] - P.max()) + 255

    return Q.round()


def whiten(P):
    Q = np.empty(P.shape)
    i, j = P.shape
    for a in range(i):
        for b in range(j):
            Q[a, b] = (P[a, b] - P.mean()) / P.std()

    return Q
