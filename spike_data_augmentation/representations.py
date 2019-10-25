import numpy as np


class Timesurface(object):
    """Representation that creates timesurfaces for each event for one recording.

    Args:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        array of timesurfaces with dimensions (w,h)
    """

    def __init__(
        self, surface_dimensions=(7, 7), tau=5e3, decay="lin", merge_polarities=False
    ):
        assert len(surface_dimensions) == 2
        assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
        self.surface_dimensions = surface_dimensions
        self.radius_x = surface_dimensions[0] // 2
        self.radius_y = surface_dimensions[1] // 2
        self.tau = tau
        self.decay = decay
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None):
        assert "x" and "y" and "t" and "p" in ordering
        assert len(sensor_size) == 2
        x_index = ordering.find("x")
        y_index = ordering.find("y")
        t_index = ordering.find("t")
        p_index = ordering.find("p")
        n_of_events = len(events)
        if self.merge_polarities:
            events[:, p_index] = np.zeros(n_of_events)
        n_of_pols = len(np.unique(events[:, p_index]))
        timestamp_memory = np.zeros(
            (
                n_of_pols,
                sensor_size[0] + self.radius_x * 2,
                sensor_size[1] + self.radius_y * 2,
            )
        )
        timestamp_memory -= self.tau * 3 + 1
        all_surfaces = np.zeros(
            (
                n_of_events,
                n_of_pols,
                self.surface_dimensions[0],
                self.surface_dimensions[1],
            )
        )
        for index, event in enumerate(events):
            x = int(event[x_index])
            y = int(event[y_index])
            timestamp_memory[
                int(event[p_index]), x + self.radius_x, y + self.radius_y
            ] = event[t_index]
            timestamp_context = (
                timestamp_memory[
                    :,
                    x : x + self.surface_dimensions[0],
                    y : y + self.surface_dimensions[1],
                ]
                - event[t_index]
            )
            if self.decay == "lin":
                timesurface = timestamp_context / (3 * self.tau) + 1
                timesurface[timesurface < 0] = 0
            elif self.decay == "exp":
                timesurface = np.exp(timestamp_context / self.tau)
                timesurface[timestamp_context < (-3 * self.tau)] = 0
            all_surfaces[index, :, :, :] = timesurface
        return all_surfaces
