import numpy as np
import warnings


def time_skew_numpy(
    events: np.ndarray,
    ordering: str,
    coefficient: float,
    offset: int = 0,
    integer_time: bool = False,
):
    """Skew all event timestamps according to a linear transform,
    potentially sampled from a distribution of acceptable functions.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels].
        ordering: ordering of the event tuple inside of events. This function requires 't'
                  to be in the ordering
        coefficient: a real-valued multiplier applied to the timestamps of the events.
                     E.g. a coefficient of 2.0 will double the effective delay between any
                     pair of events.
        offset: value by which the timestamps will be shifted after multiplication by
                the coefficient. Negative offsets are permissible but may result in
                in an exception if timestamps are shifted below 0.
        integer_time: flag that specifies if timestamps should be rounded to
                             nearest integer after skewing.

    Returns:
        the input events with rewritten timestamps.
    """

    assert "t" in ordering
    t_index = ordering.index("t")

    events[:, t_index] = events[:, t_index] * coefficient + offset

    if integer_time:
        events[:, t_index] = events[:, t_index].round()

    return events
