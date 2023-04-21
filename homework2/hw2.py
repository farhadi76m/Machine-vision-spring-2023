import os

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv

from utils import vis
from utils import filters
from time import time

if __name__ == "__main__":
    path = os.listdir('.')
    images = []
    for item in path:
        if item.split('.')[1] == 'bmp':
            image = Image.open(item)
            image = np.array(image)
            images.append(image)
    vis.show_grays(images)

    # Decomposition Image
    decompose = {'image': [],
                 'name': []}
    name = ['rgb image', 'red channel', 'green channel', 'blue channel']
    for image in images:
        r = image * np.array([1, 0, 0])
        g = image * np.array([0, 1, 0])
        b = image * np.array([0, 0, 1])

        _dec = [image, r, g, b]
        decompose['image'] += _dec
        decompose['name'] += name
    vis.show_grays(decompose, 4)

    #
    hsv_images = {'image': [],
                  'name': []}
    for image in images:
        hsv = rgb_to_hsv(image / 255)
        hsv_images['image'] += [image, hsv]
        hsv_images['name'] += ['rgb', 'hsv']
    vis.show_grays(hsv_images, 4)
    #

    y = lambda a: 1 / 2 * a ** 3
    linear_filter = {'image': [],
                     'name': []}
    for image in images:
        image = image / 255
        _image = y(image)
        linear_filter['image'] += [image, _image]
        linear_filter['name'] += ['before', 'after']
    vis.show_grays(linear_filter, 4)

    # Zero Pad
    padded_images = {'image': [],
                     'name': []}
    for image in images:
        h = np.random.randint(10)
        w = np.random.randint(10)
        padded = filters.zero_pad(image, h, w)
        padded_images['image'] += [image, padded]
        padded_images['name'] += ['without pad', f'after padding h:{h}, w:{w}']
    vis.show_grays(padded_images)
    #  Conv
    # Simple convolution kernel.
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])
    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    time1 = time()
    conv = [filters.conv(test_img, kernel) for i in range(1000)]
    time1 = time() - time1

    time2 = time()
    conv2 = [filters.conv_fast(test_img, kernel) for i in range(1000)]
    time2 = time() - time2

    time3 = time()
    conv3 = [filters.conv_faster(test_img, kernel) for i in range(1000)]
    time3 = time() - time3

    print(f"simple conv take avg :{time1:.2f} s, conv fast take avg :{time2:.2f}, conv faster take avg :{time3:.2f}")

    convs = [test_img, conv[0], conv2[0], conv3[0]]

    vis.show_grays(convs)

    #
    cross_corr = filters.cross_correlation(test_img, kernel)
    cross = [test_img, kernel, cross_corr, conv[0]]

    vis.show_grays(cross)

    #
