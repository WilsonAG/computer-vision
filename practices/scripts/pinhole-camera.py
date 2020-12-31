import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [3, 8, 12],
    [2, 3, 13],
    [2, 6, 10],
    [2, 3, 7],
    [7, 3, 13],
    [7, 6, 10],
    [7, 3, 7],
])

phi = np.array([[36, 0], [0, 24]])


def get_xy(data, focal_distance):
    new_points = []
    for coord in data:
        u, v, w = coord
        point = 1/w * np.dot(focal_distance, np.array([u, v]).T)
        new_points.append(point)
    return new_points

print(get_xy(data, phi))