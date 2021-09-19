import numpy as np

from .utils import is_multi_image


def flip_ud_numpy(
    events, sensor_size, ordering, flip_probability=0.5,
):
    """
    Flips events in y. Pixels map as:

        y' = height - y

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events. This function requires 'x'
                  to be in the ordering
        flip_probability: probability of performing the flip

    Returns:
        - returns every event with y' = (sensor_size[1] - 1) - y
    """

    assert "y" in ordering

    if np.random.rand() <= flip_probability:
        y_loc = ordering.index("y")

        events[:, y_loc] = sensor_size[1] - 1 - events[:, y_loc]

    return events, sensor_size
