import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [2, 0, 10],
    [2, 3, 13],
    [2, 6, 10],
    [2, 3, 7],
    [7, 3, 13],
    [7, 6, 10],
    [7, 3, 7],
])

phi = np.array([[20, 0], [0, 18]])


def get_xy(data, focal_distance):
    new_points = []
    for coord in data:
        u, v, w = coord
        point = 1/w * np.dot(focal_distance, np.array([u, v]).T)
        new_points.append(point)
    return new_points
