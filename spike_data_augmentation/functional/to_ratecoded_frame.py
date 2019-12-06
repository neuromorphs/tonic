import numpy as np
import sparse


def to_ratecoded_frame_numpy(events, sensor_size, ordering, frame_time=5000):
    """Representation that creates frames by encoding the rate of events.

    Args:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        array of timesurfaces with dimensions (w,h)
    """
    assert "x" and "y" and "t" and "p" in ordering
    assert len(sensor_size) == 2
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_events = len(events)

    events[:, p_index][events[:, p_index] == -1] = 0

    frame_index_borders = []
    n = 1
    for i, t in enumerate(events[:, t_index]):
        if t >= frame_time * n:
            frame_index_borders.append(i - 1)
            n += 1
    frame_events = np.split(events, frame_index_borders)

    frames = np.zeros((len(frame_events) + 2,) + sensor_size)
    f = 0
    for e in frame_events:
        sparse_events = sparse.COO(
            (
                e[:, p_index],
                (
                    e[:, t_index].astype(int),
                    e[:, x_index].astype(int),
                    e[:, y_index].astype(int),
                ),
            )
        ).todense()
        frames[f, :, :] = np.sum(sparse_events, axis=0)
        f += 1

    return frames
