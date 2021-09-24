import numpy as np


def sensor_size_from_events(events: np.ndarray, ordering: str):
    """
    Given events that contain 'x' and 'y' channels, return the maximum value for each plus one as integers.
    """
    assert "x" and "y" in ordering
    x_index = ordering.index("x")
    y_index = ordering.index("y")

    x_max = int(events['x'].max() + 1)
    y_max = int(events['y'].max() + 1)
    return x_max, y_max


def is_multi_image(images):
    pass
