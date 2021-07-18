import numpy as np


def time_jitter_numpy(
    events: np.ndarray,
    ordering: str,
    std: float = 1,
    integer_jitter: bool = False,
    clip_negative: bool = False,
    sort_timestamps: bool = False,
):
    """Changes timestamp for each event by drawing samples from a Gaussian
    distribution and adding them to each timestamp.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events. This function requires 't'
                  to be in the ordering
        std: the standard deviation of the time jitter
        integer_jitter: will round the jitter that is added to timestamps
        clip_negative: drops events that have negative timestamps
        sort_timestamps: sort the events by timestamps after jittering

    Returns:
        temporally jittered set of events.
    """

    assert "t" in ordering

    t_index = ordering.find("t")
    shifts = np.random.normal(0, std, len(events))

    if integer_jitter:
        shifts = shifts.round()

    events[:, t_index] = events[:, t_index] + shifts

    if clip_negative:
        events = np.delete(events, (np.where(events[:, t_index] < 0)), axis=0)

    if sort_timestamps:
        events = events[np.argsort(events[:, t_index]), :]

    return events
