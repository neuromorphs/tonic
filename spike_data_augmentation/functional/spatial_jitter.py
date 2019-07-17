import numpy as np

from .utils import guess_event_ordering_numpy


def spatial_jitter_numpy(
    events,
    sensor_size=(346, 260),
    ordering=None,
    variance_x=1,
    variance_y=1,
    sigma_x_y=0,
):
    """
    Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:
        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Arguments:
    - events - ndarray of shape [num_events, num_event_channels]
    - sensor_size - size of the sensor that was used [W,H]
    - ordering - ordering of the event tuple inside of events, if None
                 the system will take a guess through
                 guess_event_ordering_numpy. This function requires 'x'
                 and 'y' to be in the ordering
    - variance_x - squared sigma value for the distribution in the x direction
    - variance_y - squared sigma value for the distribution in the y direction
    - sigma_x_y - changes skewness of distribution, only change if you want
                shifts along diagonal axis.

    Returns:
    - events - returns spatially jittered set of events
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
        assert "x" and "y" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")

    for event in events:
        event[x_index], event[y_index] = np.random.multivariate_normal(
            [event[x_index], event[y_index]],
            [[variance_x, sigma_x_y], [sigma_x_y, variance_y]],
        )

    return events
