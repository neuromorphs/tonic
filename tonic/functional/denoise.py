import numpy as np


def denoise_numpy(events, filter_time=10000):
    """Drops events that are 'not sufficiently connected to other events in the recording.'
    In practise that means that an event is dropped if no other event occured within a spatial neighbourhood
    of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to filter noisy recordings
    where events occur isolated in time.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        filter_time: maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.

    Returns:
        filtered set of events.
    """

    assert "x" and "y" and "t" in events.dtype.names

    events_copy = np.zeros_like(events)
    copy_index = 0
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    timestamp_memory = np.zeros((width, height)) + filter_time

    for event in events:
        x = int(event["x"])
        y = int(event["y"])
        t = event["t"]
        timestamp_memory[x, y] = t + filter_time
        if (
            (x > 0 and timestamp_memory[x - 1, y] > t)
            or (x < width - 1 and timestamp_memory[x + 1, y] > t)
            or (y > 0 and timestamp_memory[x, y - 1] > t)
            or (y < height - 1 and timestamp_memory[x, y + 1] > t)
        ):
            events_copy[copy_index] = event
            copy_index += 1

    return events_copy[:copy_index]
