import cv2
import numpy as np
import matplotlib.pyplot as plt

P = np.array([[5, 1, 1, 3, 2, 3],
              [6, 4, 0, 2, 3, 2],
              [7, 7, 0, 7, 3, 1],
              [9, 7, 0, 1, 1, 1],
              [8, 8, 0, 1, 1, 0],
              [7, 7, 0, 1, 0, 0]])

F1 = np.array([[1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9]])

F2 = np.array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])


def myFilter(N, F):
    F = np.flip(F)
    px = np.multiply(F, N).sum()
    return round(px)


def myConvolution(P, F):
    X = np.empty(P.shape)
    P_padd = np.pad(P, 2)
    rows, cols = P.shape
    for i in range(2, rows + 2):
        for j in range(2, cols + 2):
            N = P_padd[i - 2: i + 3, j - 2:j + 3]
            X[i - 2, j - 2] = myFilter(N, F)
    return np.uint8(X)


def myConvolution2(P, F):
    n_filas = np.size(P, 0)
    n_columnas = np.size(P, 1)
    x = np.zeros(np.shape(P))
    cont_fila = 1
    cont_col = 1
    # np.flip(F)

    for i in range(n_filas):
        for j in range(n_columnas):
            if (i == 0 or i == n_filas - 1 or j == 0 or j == n_columnas - 1):
                x[i][j] = P[i][j]
            else:
                t = np.multiply(P[i - 1:i + 2, j - 1:j + 2], np.flip(F))
                x[i][j] = round(np.sum(t))
    return x


def kernel5x5(sigma):
    m = 1 / ((2 * np.pi) * sigma ** 2)
    a = np.exp(-4 / (sigma ** 2))
    b = np.exp(-5 / (2 * sigma ** 2))
    c = np.exp(-2 / (sigma ** 2))
    d = np.exp(-1 / (sigma ** 2))
    e = np.exp(-1 / (2 * sigma ** 2))

    F = np.array([
        [a, b, c, b, a],
        [b, d, e, d, b],
        [c, e, 1, e, c],
        [b, d, e, d, b],
        [a, b, c, b, a],
    ])

    return m * F


# print(myConvolution2(P, F2))

carros_copy = cv2.imread("/home/will/Documentos/computer-vision/relay/pr7-images/carros.jpg")
carros_copy = cv2.cvtColor(carros_copy, cv2.COLOR_BGR2RGB)
carros = cv2.imread("/home/will/Documentos/computer-vision/relay/pr7-images/carros.jpg", cv2.IMREAD_GRAYSCALE)
width = int(carros.shape[1] * 10 / 100)
height = int(carros.shape[0] * 10 / 100)
dim = (width, height)

carros = cv2.resize(carros, dim, interpolation=cv2.INTER_AREA)
carros_copy = cv2.resize(carros_copy, dim, interpolation=cv2.INTER_AREA)
F_g1 = kernel5x5(1)
F_g2 = kernel5x5(3)
F_g3 = kernel5x5(5)

X1 = myConvolution(carros, F_g1)
X2 = myConvolution(carros, F_g2)
X3 = myConvolution(carros, F_g3)

plt.title('Imagen original')
plt.imshow(carros, cmap='gray')
plt.show()

plt.title('sigma=1')
plt.imshow(X1, cmap='gray')
plt.show()

plt.title('sigma=5')
plt.imshow(X2, cmap='gray')
plt.show()

plt.title('sigma=10')
plt.imshow(X3, cmap='gray')
plt.show()

mapaBordes = cv2.Canny(carros, 200, 300)
R = carros_copy[:, :, 0]
G = carros_copy[:, :, 1]
B = carros_copy[:, :, 2]

P1_R = np.minimum(R, 255 - mapaBordes)
P1_G = np.maximum(G, mapaBordes)
P1_B = np.minimum(B, 255 - mapaBordes)
P1 = np.dstack((P1_R, P1_G, P1_B))

plt.title('Bordes canny')
plt.imshow(P1)
plt.show()

# carrorgb = cv2.cvtColor(carros, cv2.COLOR_GRAY2RGB)
#
# carrorgb[:, :, 1] = np.maximum(carrorgb[:, :, 1], mapaBordes)
# plt.imshow(carrorgb)
# plt.show()


