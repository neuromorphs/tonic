import numpy as np

import warnings

from .utils import guess_event_ordering_numpy


def time_skew_numpy(events, ordering=None, coefficient=0.9, offset=0):
    """Skew all event timestamps according to a linear transform,
    potentially sampled from a distribution of acceptable functions.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 't'
                  to be in the ordering
        coefficient: a real-valued multiplier applied to the timestamps of the events.
                     E.g. a coefficient of 2.0 will double the effective delay between any
                     pair of events.
        offset: value by which the timestamps will be shifted after multiplication by
                the coefficient. Negative offsets are permissible but may result in
                in an exception if timestamps are shifted below 0.

    Returns:
        the input events with rewritten timestamps (rounded to nearest integer if timestamps used integers in the first place.)
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "t" in ordering

    t_index = ordering.index("t")

    if np.issubdtype(events.dtype, np.integer):
        events[:, t_index] = (events[:, t_index] * coefficient + offset).round()
    else:
        events[:, t_index] = events[:, t_index] * coefficient + offset

    assert np.min(events[:, t_index]) >= 0

    return events
