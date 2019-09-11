import numpy as np

import warnings

from .utils import guess_event_ordering_numpy


def time_skew_numpy(events, ordering=None, coefficient=0.9, offset=0):
    """
    Skew all event timestamps according to a linear transform,
    potentially sampled from a distribution of acceptable functions.

    Arguments:

    events - ndarray of shape [num_events, num_event_channels]
    ordering - ordering of the event tuple inside of events, if None
    the system will take a guess through
    guess_event_ordering_numpy. This function requires 't'
    to be in the ordering
    coefficient - a real-valued multiplier applied to the timestamps of the events.
    E.g. a coefficient of 2.0 will double the effective delay between any
    pair of events.
    offset - value by which the timestamps will be shifted after multiplication by
    the coefficient. Negative offsets are permissible but may result in
    in an exception if timestamps are shifted below 0.
    Returns:

    events - returns the input events with rewritten timestamps (rounded to
             nearest integer if timestamps used integers in the first place.)
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "t" in ordering

    t_loc = ordering.index("t")

    if (events[:, t_loc] == events[:, t_loc].astype(np.int)).all():
        # timestamps have integer format (but may still be floats, e.g. 3.0)
        events[:, t_loc] = (events[:, t_loc] * coefficient + offset).round()
    else:
        events[:, t_loc] = events[:, t_loc] * coefficient + offset

    assert np.min(events[:, t_loc]) >= 0

    return events
