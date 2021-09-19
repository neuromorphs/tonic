import numpy as np

from .utils import is_multi_image


def flip_lr_numpy(
    events, sensor_size, ordering, flip_probability=0.5,
):
    """Flips events and images in x. Pixels map as:

        x' = width - x

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events. This function requires 'x'
                  to be in the ordering
        flip_probability: probability of performing the flip

    Returns:
        - every event with x' = sensor_size[1] - 1 - x
    """

    assert "x" in ordering

    if np.random.rand() <= flip_probability:
        x_loc = ordering.index("x")
        events[:, x_loc] = sensor_size[0] - 1 - events[:, x_loc]

    return events, sensor_size
