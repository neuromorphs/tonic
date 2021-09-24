import numpy as np
from typing import List


def spatial_jitter_numpy(
    events: np.ndarray,
    sensor_size: List[int],
    variance_x: float = 1,
    variance_y: float = 1,
    sigma_x_y: float = 0,
    clip_outliers: bool = False,
):
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:

        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        variance_x: squared sigma value for the distribution in the x direction
        variance_y: squared sigma value for the distribution in the y direction
        sigma_x_y: changes skewness of distribution, only change if you want shifts along diagonal axis.
        clip_outliers: when True, events that have been jittered outside the sensor size will be dropped.

    Returns:
        array of spatially jittered events.
    """

    assert "x" and "y" in events.dtype.names

    shifts = np.random.multivariate_normal(
        [0, 0], [[variance_x, sigma_x_y], [sigma_x_y, variance_y]], len(events)
    )

    events["x"] = events["x"] + shifts[:, 0]
    events["y"] = events["y"] + shifts[:, 1]

    if clip_outliers:
        events = np.delete(
            events,
            np.where(
                (events["x"] < 0)
                | (events["x"] >= sensor_size[0])
                | (events["y"] < 0)
                | (events["y"] >= sensor_size[1])
            ),
            axis=0,
        )

    return events
