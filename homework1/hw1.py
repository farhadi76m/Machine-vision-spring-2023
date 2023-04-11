import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils import transforms
from utils import vis

m = nn.UpsamplingBilinear2d(scale_factor=4)

if __name__ == "__main__":
    image = Image.open("man.bmp")
    image = np.array(image)

    QUANTIZE_LEVEL: list[int] = [64, 16, 8, 4, 2]
    q_images = transforms.quantize(image, QUANTIZE_LEVEL)

    vis.show_grays(q_images)
    breakpoint()
    RESIZE_LEVEL = [512, 256, 128, 64, 32]
    r_images = transforms.resize(image, RESIZE_LEVEL)
    vis.show_grays(r_images)

    r_images['upsample'] = []
    for img in (r_images['image']):
        scale_factor = img.shape
        print(scale_factor)
        img = img.repeat(1024 // scale_factor[0], 0).repeat(1024 // scale_factor[1], 1)
        r_images['upsample'].append(img)

    vis.show_grays(r_images['upsample'])

    r_images['bilinear'] = []
    for img in (r_images['image']):
        scale_factor = img.shape
        m = nn.UpsamplingBilinear2d(scale_factor=1024 // scale_factor[0])
        print(image.shape)
        img = m(torch.from_numpy(img).unsqueeze(0).unsqueeze(0) / 256)
        r_images['bilinear'].append(img.squeeze())

    vis.show_grays(r_images['bilinear'])
