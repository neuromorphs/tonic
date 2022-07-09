import numpy as np


def drop_event_numpy(events, drop_probability=0.5, random_drop_probability=False):
    """Randomly drops events with drop_probability.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels].
        drop_probability: probability of dropping out event.
        random_drop_probability: randomize the dropout probability
                                 between 0 and drop_probability.

    Returns:
        augmented events that were not dropped.
    """

    if random_drop_probability is True:
        drop_probability *= np.random.rand()

    length = events.shape[0]  # find the number of events
    nDrop = int(drop_probability * length + 0.5)
    ind = np.random.choice(length, nDrop, replace=False)
    return np.delete(events, ind, axis=0)


def drop_by_time_numpy(events, duration_ratio=0.2, starts_at_zero=True):
    assert "x" and "t" and "p" in events.dtype.names
    assert (
        type(duration_ratio) == float
        and duration_ratio >= 0.0
        and duration_ratio <= 1.0
    ) or (
        type(duration_ratio) == tuple
        and len(duration_ratio) == 2
        and all(val >= 0 and val <= 1.0 for val in duration_ratio)
    )

    # time interval
    t_start = 0.0 if starts_at_zero else events["t"].min()
    t_end = events["t"].max()

    if type(duration_ratio) is tuple:
        duration_ratio = np.random.uniform(duration_ratio[0], duration_ratio[1])

    drop_duration = (t_end - t_start) * duration_ratio

    drop_start = np.random.uniform(t_start, t_end - drop_duration)
    mask_events = (events["t"] >= drop_start) & (
        events["t"] <= drop_start + drop_duration
    )

    return np.delete(events, mask_events)  # remove events


def drop_by_area_numpy(events, sensor_size, area_ratio=0.2):
    assert "x" and "t" and "p" in events.dtype.names
    assert (type(area_ratio) == float and area_ratio >= 0.0 and area_ratio <= 1.0) or (
        type(area_ratio) is tuple
        and len(area_ratio) == 2
        and all(val >= 0 and val <= 1.0 for val in area_ratio)
    )

    if not sensor_size:
        sensor_size_x = int(events["x"].max() + 1)
        sensor_size_p = len(np.unique(events["p"]))
        if "y" in events.dtype.names:
            sensor_size_y = int(events["y"].max() + 1)
            sensor_size = (sensor_size_x, sensor_size_y, sensor_size_p)
        else:
            sensor_size = (sensor_size_x, 1, sensor_size_p)

    # select ratio
    if type(area_ratio) is tuple:
        area_ratio = np.random.uniform(area_ratio[0] and area_ratio[1])

    # select area
    cut_w = int(sensor_size[0] * area_ratio)
    cut_h = int(sensor_size[1] * area_ratio)
    bbx1 = np.random.randint(0, (sensor_size[0] - cut_w))
    bby1 = np.random.randint(0, (sensor_size[1] - cut_h))
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h

    # filter image
    mask_events = (
        (events["x"] >= bbx1)
        & (events["y"] >= bby1)
        & (events["x"] <= bbx2)
        & (events["y"] <= bby2)
    )

    # delete events of bbox
    return np.delete(events, mask_events)  # remove events
