import numpy as np

from .utils import guess_event_ordering_numpy, is_multi_image


def drop_event_numpy(events, drop_probability=0.5):
    """
    Randomly drops events with drop_probability.

    Arguments:
    - events - ndarray of shape [num_events, num_event_channels]
    - drop_probability - probability of dropping out event

    Returns:
    - events - returns events that were not dropped
    """

    # length = events.shape[0]
    # nDrop = int(p * length)
    # ind = np.random.randint(0, length, size=nDrop)
    # return np.delete(events, ind, axis=1)
