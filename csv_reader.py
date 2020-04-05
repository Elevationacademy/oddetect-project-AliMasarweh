import pandas as pd
import os
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

__csv_reader_file_headers_glob__ = ["# Frame", " x(0-1024)", " y(0-1024)", " obj id", " bounding size(0-1024^2)",
                                    " sequence(may be normalized)", " num objects", " current_time", " current_milli"]

__csv_reader_header_glob__ = ["frame", "x", "y", "obj_id", "bounding_size",
                              "sequence", "num_objects", "current_time", "current_milli"]

color_list = [[0, 0, 0, 255], [255, 255, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255], [0, 0, 255, 255]]


def yield_file_from_sorted_directory(path):
    for root, dirs, files in os.walk(path):
        files.sort()
        yield from files


def read_csv(file, file_headers=None, header=None):
    if header is None:
        header = __csv_reader_header_glob__
    if file_headers is None:
        file_headers = __csv_reader_file_headers_glob__

    with open(file, newline='') as csv_file:
        data_frame = pd.read_csv(csv_file, header=3, usecols=file_headers)
        data_frame.columns = header

    return data_frame


def plot_filter_over_image(image, is_x_y_normalized=True):
    max_x, max_y = 1024, 1024
    img_shape_x, img_shape_y = image.shape[0], image.shape[1]

    directory = yield_file_from_sorted_directory("HUJI")
    csv_file = next(directory)
    df = read_csv("HUJI/" + csv_file)
    print(image.shape)

    for index, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if is_x_y_normalized:
            x, y = int((x / max_x) * img_shape_x), int((y / max_y) * img_shape_y)

        obj_id = int(row["obj_id"])

        obj_id = obj_id % 5
        image[x - 2:x + 2, y - 2:y + 2] = np.asarray(color_list[obj_id], dtype=int)

    plt.imshow(image)
    plt.show()


def plot_filtering_matrix_over_image(image, is_x_y_normalized=True):
    matrix, mask = np.zeros(image.shape, dtype=int), np.ones(image.shape, dtype=bool)
    max_x, max_y = 1024, 1024
    img_shape_x, img_shape_y = image.shape[0], image.shape[1]

    directory = yield_file_from_sorted_directory("HUJI")
    csv_file = next(directory)
    df = read_csv("HUJI/" + csv_file)
    print(image.shape)

    for index, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if is_x_y_normalized:
            x, y = int((x / max_x) * img_shape_x), int((y / max_y) * img_shape_y)

        obj_id = int(row["obj_id"])

        obj_id = obj_id % 5
        matrix[x - 2:x + 2, y - 2:y + 2] = np.asarray(color_list[obj_id], dtype=int)
        mask[x - 2:x + 2, y - 2:y + 2] = False

    matrix[mask] = image[mask]
    plt.imshow(matrix)
    plt.show()


if __name__ == "__main__":
    image = imread("base.png")
    plot_filtering_matrix_over_image(image)
