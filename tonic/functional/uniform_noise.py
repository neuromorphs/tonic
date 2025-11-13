import numpy as np


def uniform_noise_numpy(events: np.ndarray, sensor_size: tuple[int, int, int], n: int):
    """Adds a fixed number of noise events that are uniformly distributed across sensor size
    dimensions.

    Parameters:
        events: ndarray of shape (n_events, n_event_channels)
        sensor_size: 3-tuple of integers for x, y, p
        n: the number of noise events added.
    """
    noise_events = np.zeros(n, dtype=events.dtype)
    for channel in events.dtype.names:
        if channel == "x":
            noise_events[channel] = np.random.randint(0, sensor_size[0], size=n)
        if channel == "y":
            noise_events[channel] = np.random.randint(0, sensor_size[1], size=n)
        if channel == "p":
            noise_events[channel] = np.random.randint(0, sensor_size[2], size=n)
        if channel == "t":
            t_min, t_max = events["t"].min(), events["t"].max()
            noise_events[channel] = np.random.uniform(t_min, t_max, size=n)

    noisy_events = np.concatenate((events, noise_events))
    return noisy_events[np.argsort(noisy_events["t"])]
