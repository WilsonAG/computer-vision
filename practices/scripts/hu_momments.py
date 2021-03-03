import numpy as np


def moment(p, q, P):
    rows, cols = P.shape
    sumatory = 0
    for i in range(rows):
        for j in range(cols):
            # print(pow(i + 1, p) * pow(j + 1, q) * P[i, j], end=' ')
            sumatory += pow(i + 1, p) * pow(j + 1, q) * P[i, j]
    return sumatory


def central_moment(p, q, P):
    m00 = moment(0, 0, P)
    x = moment(1, 0, P) / m00
    y = moment(0, 1, P) / m00
    rows, cols = P.shape
    sumatory = 0
    for i in range(rows):
        for j in range(cols):
            # print(pow(i + 1 - x, p) * pow(j + 1 - y, q) * P[i, j], end=' ')
            sumatory += pow(i + 1 - x, p) * pow(j + 1 - y, q) * P[i, j]
    return sumatory


def compute_hu_moments(P):
    return np.array([
        round(central_moment(2, 0, P) + central_moment(0, 2, P), 2),
        round((central_moment(2, 0, P) - central_moment(0, 2, P)) ** 2 + 4 * central_moment(1, 1, P) ** 2, 2)
    ])


__P = np.array([
    [30, 0],
    [250, 0],
    [28, 120]
])
cm = central_moment(2, 0, __P)
print(cm)
