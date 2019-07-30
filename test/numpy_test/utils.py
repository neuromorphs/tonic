import numpy as np


def create_random_input_with_ordering(ordering):
    """
    Creates a random frame to use for tests with certain ordering

    Returns
    - events - 10k events in xytp formatting
    - images - 4 images at sensor_size
    - sensor_size - 200 x 100 (w,h)
    - ordering as input
    - multi_image - True
    """
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")

    sensor_size = (200, 100)  # width x height
    events = np.random.rand(10000, 4)

    events[:, x_index] = (events[:, x_index] * sensor_size[0]).astype(int)
    events[:, y_index] = (events[:, y_index] * sensor_size[1]).astype(int)
    events[events[:, p_index] < 0.5, p_index] = -1
    events[events[:, p_index] >= 0.5, p_index] = 1

    # sort timestamps to ensure the times are sequential
    events[:, t_index] = np.sort(events[:, t_index])
    images = np.random.rand(4, sensor_size[1], sensor_size[0])

    return events, images, sensor_size, ordering, True
