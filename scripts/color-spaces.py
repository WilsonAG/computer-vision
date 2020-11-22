import numpy as np
import collections


def get_hitogram(mtx):
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
    return nc / (i*j)


Q = np.array([
    [0, 0, 51, 204, 204, 255],
    [0, 51, 153, 204, 153, 204],
    [51, 153, 204, 102, 51, 153],
    [204, 204, 153, 51, 0, 0],
    [255, 204, 102, 51, 0, 0],
    [255, 255, 204, 153, 51, 0],
])

P = np.array([
    [150, 210, 210, 210, 150],
    [50, 0, 50, 180, 255],
    [190, 0, 180, 190, 180],
    [150, 210, 190, 180, 150],
])

phi = 180

hist = get_hitogram(P)

hist_blk, mean_black = get_mean(hist, phi, 0)
hist_wh, mean_white = get_mean(hist, phi, 1)

var_black = get_variance(hist_blk, mean_black)
var_white = get_variance(hist_wh, mean_white)

w_blk = get_weight(hist_blk, P.shape)
w_wh = get_weight(hist_wh, P.shape)


ic_var = w_blk * var_black + w_wh * var_white

