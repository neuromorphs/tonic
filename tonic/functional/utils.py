import warnings
import numpy as np


def is_multi_image(images, sensor_size):
    """
    Guesses at if there are multiple images inside of images

    Arguments:
    - images - image array to find where sensor_size is supported shapes
               include
               - [num_images, height, width, num_channels]
               - [height, width, num_channels]
               - [num_images, height, width]
               - [height, width]
    - sensor_size - sensor [W,H]

    Returns:
    - guess - best guess at if there are multiple images
    """

    warnings.warn("[Tonic]::Guessing if there are multiple images")
    if len(images.shape) == 4:
        guess = True
    elif len(images.shape) == 3:
        if images.shape[0] == sensor_size[0]:
            guess = False  # HWC
        else:
            guess = True  # NHW
    elif len(images.shape) == 2:
        guess = False
    else:
        raise NotImplementedError()
    warnings.warn("[Tonic]::Guessed [%s]" % str(guess))

    return guess
