import numpy as np


def refractory_period_numpy(events, refractory_period=10000):
    """Sets a refractory period for each pixel, during which events will be
    ignored/discarded. We keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        refractory_period: refractory period for each pixel in microseconds

    Returns:
        filtered set of events.
    """

    assert "t" and "x" and "y" in events.dtype.names

    events_copy = np.zeros_like(events)
    copy_index = 0
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    timestamp_memory = np.zeros((width, height)) - refractory_period

    for event in events:
        time_since_last_spike = (
            event["t"] - timestamp_memory[int(event["x"]), int(event["y"])]
        )
        if time_since_last_spike > refractory_period:
            events_copy[copy_index] = event
            copy_index += 1
        timestamp_memory[int(event["x"]), int(event["y"])] = event["t"]

    return events_copy[:copy_index]
