import numpy as np


def create_random_input_xytp():
    """
    Creates a random frame to use for tests

    Returns

    - events - 10k events in xytp formatting
    - images - 4 images at sensor_size
    - sensor_size - 200 x 100 (w,h)
    - ordering - xytp
    - multi_image - True
    """

    sensor_size = (200, 100)  # width x height

    events = np.random.rand(10000, 4)

    events[:, 0] *= sensor_size[0]
    events[:, 1] *= sensor_size[1]

    events[events[:, 3] < 0.5, 3] = -1
    events[events[:, 3] >= 0.5, 3] = 1

    # sort timestamps to ensure the times are sequential
    events[:, 2] = np.sort(events[:, 2])

    images = np.random.rand(4, sensor_size[1], sensor_size[0])

    return events, images, sensor_size, "xytp", True
