import numpy as np


def create_random_input(
    sensor_size=(200, 100, 2),
    n_events=10000,
    dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]),
):
    """Creates random events for testing purposes.

    Returns
    - events - 10k events
    - sensor_size - 200 x 100 x 2 (w,h,pol)
    """

    assert "x" and "t" and "p" in dtype.names

    events = np.zeros(n_events, dtype=dtype)
    events["x"] = np.random.rand(n_events) * sensor_size[0]
    events["p"] = np.random.rand(n_events) * sensor_size[2]
    # sort timestamps to ensure the times are sequential
    events["t"] = np.sort(np.random.rand(n_events) * 1e6)

    if "y" in dtype.names:
        events["y"] = np.random.rand(n_events) * sensor_size[1]

    return events, sensor_size
