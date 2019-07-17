import warnings
import numpy as np


def guess_event_ordering_numpy(events):
    """
    Guesses the names of the channels for events or returns numpy ndarray names

    Arguments:
    - events - the events in [num_events, channels]

    Returns:
    - guess - string representation of ordering of channels
    """

    warnings.warn("[SDAug]::Guessing the ordering of xytp in events")

    if np.issubdtype(events.dtype, np.number):
        if events.shape[1] == 3:
            guess = "xtp"
        elif events.shape[1] == 4:
            guess = "xytp"
        elif events.shape[1] == 5:
            guess = "xyztp"
    elif isinstance(events, (np.ndarray, np.generic)):
        guess = events.dtype.names
    else:
        raise NotImplementedError("Unable to guess event ordering")

    warnings.warn("[SDAug]::Guessed [%s] as ordering of events" % guess)

    return guess


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

    warnings.warn("[SDAug]::Guessing if there are multiple images")
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
    warnings.warn("[SDAug]::Guessed [%s]" % str(guess))

    return guess
