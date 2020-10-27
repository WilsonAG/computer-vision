import numpy as np


def get_xy_points(coords_uvw, df):
    u, v, w = coords_uvw
    fx, fy = df

    param_df = np.array([[fx, 0], [0, fy]])
    x, y = 1/w * param_df.dot([u, v])

    return tuple((round(x, 2), round(y, 2)))


coords = []
coords.append(get_xy_points((2, 0, 10), (20, 18)))
coords.append(get_xy_points((2, 3, 13), (20, 18)))
coords.append(get_xy_points((2, 6, 10), (20, 18)))
coords.append(get_xy_points((2, 3, 7), (20, 18)))
coords.append(get_xy_points((7, 0, 10), (20, 18)))
coords.append(get_xy_points((7, 3, 13), (20, 18)))
coords.append(get_xy_points((7, 6, 10), (20, 18)))
coords.append(get_xy_points((7, 3, 7), (20, 18)))

coords2 = []
coords2.append(get_xy_points((2, 1, 9), (20, 18)))
coords2.append(get_xy_points((2, 3, 11), (20, 18)))
coords2.append(get_xy_points((2, 5, 9), (20, 18)))
coords2.append(get_xy_points((2, 3, 7), (20, 18)))

coords2.append(get_xy_points((5, 1, 9), (20, 18)))
coords2.append(get_xy_points((5, 3, 11), (20, 18)))
coords2.append(get_xy_points((5, 5, 9), (20, 18)))
coords2.append(get_xy_points((5, 3, 7), (20, 18)))


print(coords2)
