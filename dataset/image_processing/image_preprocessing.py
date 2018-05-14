import numpy as np


def normalize_to_pixel_space(X):
    """
    Normalizes to pixel space no matter what the current transformation is
    X = 4d tensor of images (one channel only)
    im_ordering = n x channels x width x height
    :param X:
    :return:
    """
    # if < 0, shift to positive space
    if np.min(X) < 0:
        mins = np.min(X, axis=(1, 2, 3))
        for i in range(len(X)):
            X[i] += abs(mins[i])

    # if > 1 normalize bn 0,1
    if np.max(X) > 1:
        maxs = np.max(X, axis=(1, 2, 3))
        for i in range(len(X)):
            X[i] /= maxs[i]

    # scale to 255.0
    X *= 255.0
    return X


def crop_image_batch(images, im_size):
    height = len(images[1])
    width = len(images[2])
    x = (width - im_size) / 2
    y = (height - im_size) / 2

    images = images[:, y: y+im_size, x: x+im_size]
    return images

def crop_image(image, im_size):
    height = len(image)
    width = len(image[0])
    x = (width - im_size) / 2
    y = (height - im_size) / 2

    sub_image = image[y: y+im_size, x: x+im_size]
    return sub_image


def normalize_scale(images, mean):
    images /= 255.0
    images -= mean


def calculate_cropping_idxs(original_width, crop_width):
    """
    Assumes a square image

    :param original_width:
    :param crop_width:
    :return:
    """

    x = y = int((original_width - crop_width) / 2)
    return y, y + crop_width, x, x + crop_width


def concat_masks(data, masks_path):
    """
    Returns a 2D matrix of dim batch_size x (dim(data) + dim(flattened mask)), whose
    rows are the receptive fields mask to be concatenated with other
    inputs to networks, and dim(flattened mask)

    :param data: data to which masks are concatenated to each row
    :param masks_path: Path of receptive field masks saved as .npy file
    """
    n = data.shape[0]
    loaded_masks = np.load(masks_path)
    masks_matrix = np.array([loaded_masks for i in range(n)], dtype=np.float32)
    return np.concatenate((data, masks_matrix), axis=1)
