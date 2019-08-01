import numpy as np

from .utils import guess_event_ordering_numpy


def time_jitter_numpy(events, sensor_size=(346, 260), ordering=None, variance=1):
    """
    Changes timestamp for each event by drawing samples from a
    Gaussian distribution with the following properties:
        mean = [t]
        variance = variance

    Arguments:
    - events - ndarray of shape [num_events, num_event_channels]
    - sensor_size - size of the sensor that was used [W,H]
    - ordering - ordering of the event tuple inside of events, if None
                 the system will take a guess through
                 guess_event_ordering_numpy. This function requires 't'
                 to be in the ordering
    - variance - change the variance of the time jitter

    Returns:
    - events - returns temporally jittered set of events
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
        assert "t" in ordering

    t_index = ordering.find("t")
    shifts = np.random.normal(0, variance, len(events))
    events[:, t_index] += shifts

    return events
