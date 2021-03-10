from skimage import measure
import numpy as np
from Defines import background_size


def make_segmented_image(image, object_label):
    if image.shape == (background_size, background_size, 1):  # remove channels
        image = image.reshape((background_size, background_size))
    x_size, y_size = image.shape
    filled_array = np.full((x_size, y_size), object_label)
    segmented_array = (image == filled_array) * 1
    return segmented_array


# segments the input image with multiple objects into seperate images, each with one object
def segment(image, flatten=True):
    if image.shape == (background_size ** 2,):  # unflatten
        image = image.reshape(background_size, background_size)

    labelled_image = measure.label(image, connectivity=1)  # labels connected pieces in the image
    unique = np.unique(labelled_image)[1:]  # exclude the 0, background, label
    if unique.size == 0:
        raise ValueError('Empty image, no objects are found by skimage.measure.label')

    seg_image_list = []
    for label in unique:
        seg_image = make_segmented_image(labelled_image, label)
        seg_image_list.append(seg_image)

    if flatten:
        return np.array(seg_image_list).reshape(-1, background_size ** 2)
    else:
        return np.array(seg_image_list).reshape((-1, 1, background_size, background_size, 1))  # CNN expects [batch, im_x, im_y, channels]
