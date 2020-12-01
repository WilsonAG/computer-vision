import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [2, 0, 10],
    [2, 3, 13],
    [2, 6, 10],
    [2, 3, 7],
    [7, 0, 10],
    [7, 3, 13],
    [7, 6, 10],
    [7, 3, 7]])

phi = np.array([
    [20, 0],
    [0, 18]])

xy = np.zeros(shape=(8, 2))

j = 0
for i in A:
    u, v, w = i
    r1 = np.dot(phi, np.array([[u], [v]]))
    r2 = (1 / w) * r1
    xy[j] = [r2[0], -r2[1]]
    j = j+1

plt.scatter(xy[:, 0], xy[:, 1])

joined_indexes = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]

xy_join = xy[joined_indexes, :]

plt.plot(xy_join[:, 0], xy_join[:, 1])
# print(xy_join)
plt.show()
