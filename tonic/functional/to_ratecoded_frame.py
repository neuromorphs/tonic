import numpy as np
import math


def to_ratecoded_frame_numpy(
    events, sensor_size, ordering, frame_time=5000, merge_polarities=True
):
    """Representation that creates frames by encoding the rate of events.

    Args:
        frame_time: time bin for each frame
        merge_polarities: flag to add all polarities together

    Returns:
        numpy array of n rate-coded frames (n,w,h)
    """
    assert "x" and "y" and "t" and "p" in ordering
    assert len(sensor_size) == 2
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_events = len(events)

    pols = events[:, p_index]
    if merge_polarities:
        pols[pols == -1] = 1
        pols[pols == 0] = 1
    else:
        pols[pols == -1] = 0

    n_bins = math.ceil(events[-1, t_index] / frame_time)

    n = 0
    frames = np.zeros((n_bins, sensor_size[1], sensor_size[0]))
    for e in events:
        x = int(e[x_index])
        y = int(e[y_index])
        t = e[t_index]
        p = e[p_index]

        if t > frame_time * (n + 1):
            n += 1
        frames[n, y, x] += p

    frames = np.tanh(frames / 3) * 255

    return frames
