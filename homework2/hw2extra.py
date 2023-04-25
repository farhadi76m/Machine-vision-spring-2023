import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import vis, transforms
from utils import filters
from time import time

print(os.getcwd())
if __name__ == "__main__":
    shelf = transforms.readImage('shelf.jpg')
    template = transforms.readImage('template.jpg')

    shelf_gray = np.sum(shelf * np.array([3, 6, 1]) / 10, 2)
    template_gray = np.sum(template * np.array([3, 6, 1]) / 10, 2)
    vis.show_grays([shelf_gray, shelf, template_gray, template])

    corr = filters.cross_correlation(shelf_gray, template_gray)
    y, x = (np.unravel_index(corr.argmax(), corr.shape))
    # Draw marker at detected location

    # Display cross-correlation output
    plt.subplot(2, 2, 1)
    plt.imshow(shelf)
    plt.title('Cross-correlation (white means more correlated)')
    plt.axis('off')

    # Display image
    plt.subplot(2, 2, 2)
    plt.imshow(corr, 'rainbow')
    plt.title('Result (blue marker on the detected location)')
    plt.axis('off')

    # Draw marker at detected location
    plt.plot(x, y, 'bo', ms=20, mew=5)
    # plt.show()

    template_gray2 = template_gray - template_gray.mean()
    corr2 = filters.cross_correlation(shelf_gray, template_gray2)
    y2, x2 = (np.unravel_index(corr2.argmax(), corr2.shape))

    # Display image
    plt.subplot(2, 2, 3)
    plt.imshow(corr2, 'rainbow')
    plt.title('Result (blue marker on the detected location)')
    plt.axis('off')

    # Draw marker at detected location
    plt.plot(x, y, 'bo', ms=20, mew=5)

    template_gray3 = (template_gray - template_gray.mean()) / np.var(template_gray)
    shelf_gray3 = (shelf_gray - shelf_gray.mean()) / np.var(shelf_gray)
    corr3 = filters.cross_correlation(shelf_gray3, template_gray3)
    y3, x3 = (np.unravel_index(corr3.argmax(), corr3.shape))

    # Display image
    plt.subplot(2, 2, 4)
    plt.imshow(corr3, 'rainbow')
    plt.title('Result ((red marker on the detected location))')
    plt.axis('off')

    # Draw marker at detected location
    plt.plot(x3, y3, 'rx', ms=20, mew=5)
    plt.show()
