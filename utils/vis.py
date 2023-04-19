import matplotlib.pylab as plt
import numpy as np


def show_grays(images, cols=2):
    plt.rcParams['figure.figsize'] = (10, 20)
    imgs = images['image'] if isinstance(images, dict) else images

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, ax = plt.subplots(ncols=cols, nrows=np.ceil(len(imgs) / cols).astype(np.int8), squeeze=False)
    for i, img in enumerate(imgs):
        ax[i // cols, i % cols].imshow(np.asarray(img), cmap='gray')
        ax[i // cols, i % cols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if isinstance(images, dict): ax[i // cols, i % cols].title.set_text(images['name'][i])
    plt.show()


def rescale(image, path):
    plt.imsave(path, image, cmap='gray')
