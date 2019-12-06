import numpy as np
import math


def to_ratecoded_frame_numpy(events, sensor_size, ordering, frame_time=5000):
    """Representation that creates frames by encoding the rate of events.

    Args:
        frame_time: time value that events should be binned into

    Returns:
        n rate-coded frames (n,w,h) 
    """
    assert "x" and "y" and "t" and "p" in ordering
    assert len(sensor_size) == 2
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_events = len(events)

    events[:, p_index][events[:, p_index] == -1] = 0
    n_bins = math.ceil(events[-1, t_index] / frame_time)

    n = 0
    frames = np.zeros((n_bins + 2,) + sensor_size)
    for i, e in enumerate(events):
        x = int(e[x_index])
        y = int(e[y_index])
        t = e[t_index]
        p = e[p_index]

        if t >= frame_time * (n + 1):
            n += 1
        frames[n, x, y] += p

    return frames
