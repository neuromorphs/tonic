from typing import List

import numpy as np


def spatial_jitter_numpy(
    events: np.ndarray,
    sensor_size: List[int],
    var_x: float = 1,
    var_y: float = 1,
    sigma_xy: float = 0,
    clip_outliers: bool = False,
):
    """Changes x/y coordinate for each event by adding samples from a multivariate Gaussian
    distribution. It with the following properties:

        .. math::
            mean = [x,y]

            \Sigma = [[var_x, sigma_{xy}],[sigma_{xy}, var_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        var_x: squared sigma value for the distribution in the x direction
        var_y: squared sigma value for the distribution in the y direction
        sigma_xy: changes skewness of distribution, only change if you want shifts along diagonal axis.
        clip_outliers: when True, events that have been jittered outside the sensor size will be dropped.

    Returns:
        array of spatially jittered events.
    """

    assert "x" and "y" in events.dtype.names

    shifts = np.random.multivariate_normal(
        [0, 0], [[var_x, sigma_xy], [sigma_xy, var_y]], len(events)
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
