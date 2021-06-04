import numpy as np


def time_jitter_numpy(
    events,
    ordering,
    std=1,
    integer_jitter=False,
    clip_negative=False,
    sort_timestamps=False,
):
    """Changes timestamp for each event by drawing samples from a
    Gaussian distribution with the following properties:

        mean = [t]
        std = std

    Will clip negative timestamps by default.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events. This function requires 't'
                  to be in the ordering
        std: change the standard deviation of the time jitter
        integer_jitter: will round the jitter that is added to timestamps
        clip_negative: drops events that have negative timestamps
        sort_timestamps: sort the events by timestamps

    Returns:
        temporally jittered set of events.
    """

    assert "t" in ordering

    t_index = ordering.find("t")
    shifts = np.random.normal(0, std, len(events))

    if integer_jitter:
        shifts = shifts.round()

    times = events[:, t_index]

    if np.issubdtype(events.dtype, np.integer):
        times += shifts.astype(np.int)
    else:
        times += shifts

    if clip_negative:
        events = np.delete(events, (np.where(times < 0)), axis=0)

    if sort_timestamps:
        events = events[np.argsort(events[:, t_index]), :]

    return events
