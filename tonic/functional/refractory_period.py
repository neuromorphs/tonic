import numpy as np


def refractory_period_numpy(events, sensor_size, ordering, refractory_period=0.5):
    """Sets a refractory period for each pixel, during which events will be
    ignored/discarded. We keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events. This function requires 't', 'x'
                  and 'y' to be in the ordering
        refractory_period: refractory period for each pixel in seconds

    Returns:
        filtered set of events.
    """

    assert "t" and "x" and "y" in ordering

    t_index = ordering.find("t")
    x_index = ordering.find("x")
    y_index = ordering.find("y")

    events_copy = np.zeros(events.shape, dtype=events.dtype)
    copy_index = 0
    timestamp_memory = (
        np.zeros((sensor_size[0], sensor_size[1]), dtype=events.dtype)
        - refractory_period
    )

    for event in events:
        time_since_last_spike = (
            event[t_index] - timestamp_memory[int(event[x_index]), int(event[y_index])]
        )
        if time_since_last_spike > refractory_period:
            events_copy[copy_index] = event
            copy_index += 1
        timestamp_memory[int(event[x_index]), int(event[y_index])] = event[t_index]

    return events_copy[:copy_index]
