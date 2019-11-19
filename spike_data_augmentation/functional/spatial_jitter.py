import numpy as np

from .utils import guess_event_ordering_numpy


def spatial_jitter_numpy(
    events, sensor_size, ordering=None, variance_x=1, variance_y=1, sigma_x_y=0
):
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:

        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Jittered events that lie outside the focal plane will be clipped.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x'
                  and 'y' to be in the ordering
        variance_x: squared sigma value for the distribution in the x direction
        variance_y: squared sigma value for the distribution in the y direction
        sigma_x_y: changes skewness of distribution, only change if you want shifts along diagonal axis.

    Returns:
        spatially jittered set of events.
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")

    shifts = np.random.multivariate_normal(
        [0, 0], [[variance_x, sigma_x_y], [sigma_x_y, variance_y]], len(events)
    )

    if np.issubdtype(events.dtype, np.integer):
        events[:, x_index] += shifts[:, 0].round().astype(np.int)
        events[:, y_index] += shifts[:, 1].round().astype(np.int)
    else:
        events[:, x_index] += shifts[:, 0]
        events[:, y_index] += shifts[:, 1]

    xs = events[:, x_index]
    ys = events[:, y_index]

    xs[xs < 0] = 0
    xs[xs > sensor_size[0]] = sensor_size[0]
    ys[ys < 0] = 0
    ys[ys > sensor_size[1]] = sensor_size[1]

    return events
