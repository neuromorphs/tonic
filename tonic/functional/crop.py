import numpy as np


def crop_numpy(events, sensor_size, ordering, target_size):
    """Crops the sensor size to a smaller sensor.

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events. This function requires 'x'
                 and 'y' to be in the ordering
        target_size: size of the sensor that was used [W',H']

    Returns:
        events - events within the crop box
        images - crop box out of the images [N,C,H,W]
    """

    assert target_size[0] <= sensor_size[0] and target_size[1] <= sensor_size[1]
    assert "x" and "y" in ordering

    x_start_ind = int(np.random.rand() * (sensor_size[0] - target_size[0]))
    y_start_ind = int(np.random.rand() * (sensor_size[1] - target_size[1]))

    x_end_ind = x_start_ind + target_size[0]
    y_end_ind = y_start_ind + target_size[1]

    x_loc = ordering.index("x")
    y_loc = ordering.index("y")

    event_mask = (
        (events[:, x_loc] >= x_start_ind)
        * (events[:, x_loc] < x_end_ind)
        * (events[:, y_loc] >= y_start_ind)
        * (events[:, y_loc] < y_end_ind)
    )

    events = events[event_mask, ...]
    events[:, x_loc] -= x_start_ind
    events[:, y_loc] -= y_start_ind

    sensor_size = target_size

    return events
