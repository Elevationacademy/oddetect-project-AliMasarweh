import numpy as np


class ImageFilterHolder:
    def __init__(self, image):
        self.image = image
        self.filter_mat = np.zeros(image.shape)

    def apply_filter(self, hours, time, coordinates, size):
        pass

    def plot_filter(self):
        pass