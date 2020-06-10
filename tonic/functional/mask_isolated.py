import numpy as np

from .utils import guess_event_ordering_numpy


def mask_isolated_numpy(
    events, sensor_size=(346, 260), ordering=None, filter_time=10000
):
    """Drops events that are 'not sufficiently connected to other events in the recording.'
    In practise that means that an event is dropped if no other event occured within a spatial neighbourhood
    of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to filter noisy recordings.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x',
                  'y' and 't' to be in the ordering
        filter_time: maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.

    Returns:
        filtered set of events.
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" and "t" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")

    events_copy = np.zeros(events.shape, dtype=events.dtype)
    copy_index = 0
    width = int(sensor_size[0])
    height = int(sensor_size[1])
    timestamp_memory = np.zeros((width, height), dtype=events.dtype) + filter_time

    for event in events:
        x = int(event[x_index])
        y = int(event[y_index])
        t = event[t_index]
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
