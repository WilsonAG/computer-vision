import numpy as np
import matplotlib.pyplot as plt


def get_histogram(mtx):
    (unique, counts) = np.unique(mtx, return_counts=True)
    freq = np.asarray((unique, counts)).T
    return freq


def get_mean(histogram, phi, color=0):
    total = 0
    div = 0
    hist_color = []
    for val, quantity in histogram:
        if color == 0 and val <= phi:
            total += val * quantity
            div += quantity
            hist_color.append([val, quantity])

        if color == 1 and val > phi:
            total += val * quantity
            div += quantity
            hist_color.append([val, quantity])

    return np.array(hist_color), total / div


def get_variance(histogram, mean, color=0):
    total = 0
    div = 0
    for val, quant in histogram:
        total += quant * pow(val - mean, 2)
        div += quant
    return total / div


def get_weight(hist_color, img_shape):
    nc = sum([x for x in hist_color[:, 1]])
    i, j = img_shape
    return nc / (i * j)


def invert_image(img):
    inverse_img = 255 - img
    inverse_img[230:235, 310:315, 1] = 255
    return inverse_img


def miRGB2gray(P, alfa, beta, gamma):
    r = P[:, :, 0]
    g = P[:, :, 1]
    b = P[:, :, 2]
    return alfa * r + beta * g + gamma * b


# No funca
def miHistograma(P):
    values, quantity = np.unique(P, return_counts=True)
    h = np.zeros((256,))

    for i in range(len(values)):
        idx = round(values[i])
        h[idx] += quantity[i]

    return h


def miGray2binaria(P, theta):
    binary = np.empty(P.shape)
    for i in range(P.shape[0]):
        binary[i, :] = [0 if val <= theta else 1 for val in P[i, :]]
    return binary


def miOtsu(P):
    var_ic = []
    hist = get_histogram(P)
    for phi in range(10, 245):
        hist_blk, mean_black = get_mean(hist, phi, 0)
        hist_wh, mean_white = get_mean(hist, phi, 1)

        var_black = get_variance(hist_blk, mean_black)
        var_white = get_variance(hist_wh, mean_white)

        w_blk = get_weight(hist_blk, P.shape)
        w_wh = get_weight(hist_wh, P.shape)

        ic_var = w_blk * var_black + w_wh * var_white
        var_ic.append(ic_var)
    # print(var_ic)
    return min(var_ic)


# _Q = np.array([
#     [0, 0, 51, 204, 204, 255],
#     [0, 51, 153, 204, 153, 204],
#     [51, 153, 204, 102, 51, 153],
#     [204, 204, 153, 51, 0, 0],
#     [255, 204, 102, 51, 0, 0],
#     [255, 255, 204, 153, 51, 0],
# ])
#
_P = np.array([
    [100,210,210,50,0,0,0,210,210],
    [0,50,0,210,1,0,1,1,0],
    [50,50,50,210,1,180,180,180,0],
    [0,50,0,1,210,0,0,0,0],
    [0,1,1,0,0,0,0,1,1]
])

_phi = 110

_hist = get_histogram(_P)

_hist_blk, _mean_black = get_mean(_hist, _phi, 0)
_hist_wh, _mean_white = get_mean(_hist, _phi, 1)

_var_black = get_variance(_hist_blk, _mean_black)
_var_white = get_variance(_hist_wh, _mean_white)

_w_blk = get_weight(_hist_blk, _P.shape)
_w_wh = get_weight(_hist_wh, _P.shape)

ic_var = _w_blk * _var_black + _w_wh * _var_white

print(_w_wh)