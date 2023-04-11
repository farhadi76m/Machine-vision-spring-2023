import matplotlib.pylab as plt
import numpy as np


def show_grays(images):
    plt.rcParams['figure.figsize'] = (10, 20)
    imgs = images['image'] if isinstance(images, dict) else images

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, ax = plt.subplots(ncols=2, nrows=len(imgs) // 2, squeeze=False)
    for i, img in enumerate(imgs):
        ax[i // 2, i % 2].imshow(np.asarray(img), cmap='gray')
        ax[i // 2, i % 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if isinstance(images, dict): ax[i // 2, i % 2].title.set_text(images['name'][i])
    plt.show()
