import numpy as np
import matplotlib.pyplot as plt


def get_a(po, pd):
    a = []

    for point in enumerate(po):
        i = point[0]
        u, v = po[i]
        x, y = pd[i]

        a.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        a.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])

    return np.array(a)


def resolve_svd(a_mtx):
    U, L, Vt = np.linalg.svd(a_mtx)
    V = np.transpose(Vt)
    phi = V[:, -1]

    phi = np.reshape(phi, [3, 3])

    return phi


def map_point(point, phi_mtx):
    u, v = point
    o = np.array([u, v, 1])

    lx, ly, l = phi_mtx.dot(o)

    # print(phi.dot(origin))

    x = int(lx / l)
    y = int(ly / l)

    return np.array([x, y])


origin = np.array([
    [0, 0],
    [300, 0],
    [300, 400],
    [0, 400]
])

dest = np.array([
    [100, 0],
    [250, 50],
    [300, 300],
    [0, 200],
])

logo = np.load('../datasets/logoUPS.npy')

A = get_a(origin, dest)
phi = resolve_svd(A)

y, x = logo.shape
new_dim = [x, y]
logo2 = np.zeros(map_point(new_dim, phi))

for i in range(x):
    for j in range(y):
        px, py = map_point([i, j], phi)
        px = int(np.round(px))
        py = int(np.round(py))
        logo2[py, px] = logo[j, i]


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(logo)
plt.subplot(1, 2, 2)
plt.imshow(logo2)
# plt.scatter(dest[0, 0], dest[0, 1])
# plt.scatter(dest[1, 0], dest[1, 1])
# plt.scatter(dest[2, 0], dest[2, 1])
# plt.scatter(dest[3, 0], dest[3, 1])
plt.show()
