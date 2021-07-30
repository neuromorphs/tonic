import numpy as np
import math


def spatial_resize_numpy(
    events: np.ndarray,
    sensor_size,
    ordering: str,
    spatial_factor: float,
    integer_coordinates: bool = False,
):
    """Resize all event x/y coordinates according to a linear transform.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels].
        ordering: ordering of the event tuple inside of events. This function requires 't'
                  to be in the ordering
        sensor_size: size of the sensor that was used [W,H]
        spatial_factor: factor to multiply each x/y coordinate with
        integer_coordinates: flag that specifies if pixel coordinates should be rounded to
                             nearest integer after resizing.

    Returns:
        the input events with resized x/y coordinates.
    """

    assert "x" and "y" in ordering

    x_index = ordering.index("x")
    y_index = ordering.index("y")

    events[:, x_index] = events[:, x_index] * spatial_factor
    events[:, y_index] = events[:, y_index] * spatial_factor

    sensor_size = [
        int(math.ceil(element * spatial_factor)) for element in list(sensor_size)
    ]

    if integer_coordinates:
        events[:, x_index] = events[:, x_index].round()
        events[:, y_index] = events[:, y_index].round()

    return events, sensor_size
