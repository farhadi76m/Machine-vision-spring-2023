import numpy as np
from PIL import Image


def readImage(image):
    image = Image.open(image)
    return np.array(image)


def quantize(image, quantize_level):
    images = {'name': [], "image": []}
    images['image'].append(image)
    images['name'].append('orginal')
    for q in quantize_level:
        bins = np.histogram_bin_edges([-1, 255], q)
        dig = np.digitize(image, bins=bins, right=True)
        images['image'].append(dig.astype(np.uint8))
        images['name'].append(f'gray level{q}')
    return images


def resize(image, resize_level):
    images = {'name': [], "image": []}
    images['image'].append(image)
    images['name'].append(str('orginal'))
    input_size = 1024
    for output_size in resize_level:
        bin_size = input_size // output_size
        small_image = image.reshape((1, output_size, bin_size,
                                     output_size, bin_size)).max(4).max(2)

        images['image'].append(small_image[0])
        images['name'].append(str(f'resize{output_size}'))
    return images

