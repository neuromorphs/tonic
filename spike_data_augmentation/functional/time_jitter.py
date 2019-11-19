import numpy as np

from .utils import guess_event_ordering_numpy


def time_jitter_numpy(events, ordering=None, variance=1):
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

    Returns:
        temporally jittered set of events.
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "t" in ordering

    t_index = ordering.find("t")
    shifts = np.random.normal(0, variance, len(events))

    if np.issubdtype(events.dtype, np.integer):
        events[:, t_index] += shifts.round().astype(np.int)
    else:
        events[:, t_index] += shifts

    times = events[:, t_index]
    times[times < 0] = 0

    return events
