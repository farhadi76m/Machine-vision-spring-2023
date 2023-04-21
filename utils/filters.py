import numpy as np


def conv(image, kernel):
    """"
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    if m + 1 - i < 0 or n + 1 - j < 0 or m + 1 - i >= Hi or n + 1 - j >= Wi:
                        sum += 0
                    else:
                        sum += kernel[i][j] * image[m + 1 - i][n + 1 - j]
            out[m][n] = sum

    return out


def zero_pad(image, pad_height, pad_width):
    """

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """
    if image.ndim == 2:
        # H, W = image.shape
        # out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
        # out[pad_height: H + pad_height, pad_width: W + pad_width] = image
        out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))
    else:
        # H, W, _ = image.shape
        # out = np.zeros((H + 2 * pad_height, W + 2 * pad_width, _))
        # out[pad_height: H + pad_height, pad_width: W + pad_width] = image
        out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)))
    return out


def conv_fast(image, kernel):
    """

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    if image.ndim == 2:

        Hi, Wi = image.shape
        Hk, Wk = kernel.shape
        out = np.zeros((Hi, Wi))
    else:
        Hi, Wi, _ = image.shape
        Hk, Wk, _ = kernel.shape
        out = np.zeros((Hi, Wi, _))

    image = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(image[m: m + Hk, n: n + Wk] * kernel)

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(np.flip(kernel, 0), 1)
    mat = np.zeros((Hi * Wi, Hk * Wk))
    for i in range(Hi * Wi):
        row = i // Wi
        col = i % Wi
        mat[i, :] = image[row: row + Hk, col: col + Wk].reshape(1, Hk * Wk)
    out = mat.dot(kernel.reshape(Hk * Wk, 1)).reshape(Hi, Wi)

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g
    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g)

    return out
