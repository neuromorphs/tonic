import numpy as np

from .utils import guess_event_ordering_numpy


def time_jitter_numpy(
    events, ordering=None, variance=1, integer_timestamps=False, clip_negative=True
):
    """Changes timestamp for each event by drawing samples from a
    Gaussian distribution with the following properties:

        mean = [t]
        variance = variance

    Automatically clipping negative timestamps.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess. This function requires 't'
                  to be in the ordering
        variance: change the variance of the time jitter
        integer_timestamps: will round the jitter that is added to timestamps
        clip_negative: drops events that have negative timestamps, otherwise set to zero.

    Returns:
        temporally jittered set of events.
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "t" in ordering

    t_index = ordering.find("t")
    shifts = np.random.normal(0, variance, len(events))

    if integer_timestamps:
        shifts = shifts.round()

    times = events[:, t_index]

    if np.issubdtype(events.dtype, np.integer):
        times += shifts.astype(np.int)
    else:
        times += shifts

    if clip_negative:
        events = np.delete(events, (np.where(times < 0)), axis=0)
    else:
        times[times < 0] = 0

    return events
