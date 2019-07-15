import numpy as np


def create_random_input_xytp():
    sensor_size = (200, 100)  # width x height

    events = np.random.rand(10000, 4)

    events[:, 0] *= sensor_size[1]
    events[:, 1] *= sensor_size[0]

    events[events[:, 3] < 0.5, 3] = -1
    events[events[:, 3] >= 0.5, 3] = 1

    # sort timestamps to ensure the times are sequential
    events[:, 2] = np.sort(events[:, 2])

    images = np.random.rand(4, sensor_size[1], sensor_size[0])

    return events, images, sensor_size, "xytp", True
