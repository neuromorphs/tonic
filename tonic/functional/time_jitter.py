import numpy as np


def time_jitter_numpy(
    events: np.ndarray,
    std: float = 1,
    clip_negative: bool = False,
    sort_timestamps: bool = False,
):
    """Changes timestamp for each event by drawing samples from a Gaussian distribution and adding
    them to each timestamp.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        std: the standard deviation of the time jitter
        clip_negative: drops events that have negative timestamps
        sort_timestamps: sort the events by timestamps after jittering

    Returns:
        temporally jittered set of events.
    """

    assert "t" in events.dtype.names

    shifts = np.random.normal(0, std, len(events))

    events["t"] = events["t"] + shifts

    if clip_negative:
        events = np.delete(events, (np.where(events["t"] < 0)), axis=0)

    if sort_timestamps:
        events = events[np.argsort(events["t"])]

    return events
