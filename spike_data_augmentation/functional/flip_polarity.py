import numpy as np

from .utils import guess_event_ordering_numpy


def flip_polarity_numpy(events, flip_probability=0.5, ordering=None):
    """Flips polarity of individual events with flip_probability.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        flip_probability: probability of flipping individual event polarities
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'p'
                  to be in the ordering and requires that polarity is
                  encoded as -1 or 1
    Returns:
        augmented events - returns every event with p' = -p at flip_probability or p' = p at 1 - flip_probability
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "p" in ordering

    p_loc = ordering.index("p")

    flips = np.ones(len(events))
    probs = np.random.rand(len(events))
    flips[probs < flip_probability] = -1

    events[:, p_loc] = events[:, p_loc] * flips

    return events
