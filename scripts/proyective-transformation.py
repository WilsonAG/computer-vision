import numpy as np

origin = np.array([
    [0, 0],
    [300, 0],
    [300, 400],
    [0, 400]
])

dest = np.array([
    [200, 0],
    [500, 100],
    [400, 400],
    [0, 200],
])


def getA(origin, dest):
    A = []

    for point in enumerate(origin):
        i = point[0]
        u, v = origin[i]
        x, y = dest[i]

        A.append([-u, -v, -1, 0, 0, 0, u*x, v*x, x])
        A.append([0, 0, 0, -u, -v, -1, u*y, v*y, y])

    return np.array(A)


def resolve_svd(A):
    U, L, Vt = np.linalg.svd(A)
    V = np.transpose(Vt)
    phi = V[:, -1]

    phi = np.reshape(phi, [3, 3])

    return phi


def map_point(point, phi):
    u, v = point
    origin = np.array([u, v, 1])

    lx, ly, l = phi.dot(origin)

    x = lx/l
    y = ly/l

    return np.array([x, y])


A = getA(origin, dest)

phi = resolve_svd(A)

p1 = [200, 120]
p2 = [300, 400]

print(p1, "-> ", map_point(p1, phi))
print(p2, "-> ", map_point(p2, phi))
