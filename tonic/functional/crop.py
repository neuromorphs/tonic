import numpy as np


def crop_numpy(events, sensor_size, target_size):
    """Crops the sensor size to a smaller sensor.

    x' = x - new_sensor_start_x
    y' = y - new_sensor_start_y

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        target_size: size of the sensor that was used [W',H']

    Returns:
        events - events within the crop box
        sensor_size - cropped to target_size
    """

    assert target_size[0] <= sensor_size[0] and target_size[1] <= sensor_size[1]
    assert "x" and "y" in events.dtype.names

    x_start_ind = int(np.random.rand() * (sensor_size[0] - target_size[0]))
    y_start_ind = int(np.random.rand() * (sensor_size[1] - target_size[1]))

    x_end_ind = x_start_ind + target_size[0]
    y_end_ind = y_start_ind + target_size[1]

    event_mask = (
        (events["x"] >= x_start_ind)
        * (events["x"] < x_end_ind)
        * (events["y"] >= y_start_ind)
        * (events["y"] < y_end_ind)
    )

    events = events[event_mask, ...]
    events["x"] -= x_start_ind
    events["y"] -= y_start_ind

    return events
