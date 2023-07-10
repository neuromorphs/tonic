from typing import Tuple

import numpy as np
from numpy.lib import recfunctions as rfn

from tonic.slicers import slice_events_by_time


def to_timesurface_numpy(
    events,
    sensor_size: Tuple[int, int, int],
    dt: float,
    tau: float,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    """Representation that creates timesurfaces for each event in the recording. Modeled after the
    paper Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern
    recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476. Unlike the paper,
    surfaces are always generated across the whole sensor, not just around the event.

    Parameters:
        sensor_size: x/y/p dimensions of the sensor
        dt: time interval at which the time-surfaces are accumulated
        tau (float): time constant to decay events around occuring event with.

    Returns:
        array of timesurfaces with dimensions (n_events//dt, p, h , w)
    """

    assert dt >= 0, print("Parameter delta_t cannot be negative.")

    event_slices = slice_events_by_time(
        events, time_window=dt, overlap=overlap, include_incomplete=include_incomplete
    )
    memory = np.zeros((sensor_size[::-1]), dtype=int)
    all_surfaces = []
    x_index = event_slices[0].dtype.names.index("x")
    y_index = event_slices[0].dtype.names.index("y")
    p_index = event_slices[0].dtype.names.index("p")
    t_index = event_slices[0].dtype.names.index("t")
    start_t = event_slices[0][0][t_index]
    for i, slice in enumerate(event_slices):
        # structured to unstructured in order to access the indices
        slice = rfn.structured_to_unstructured(slice, dtype=int)
        indices = slice[:, [p_index, y_index, x_index]].T
        timestamps = slice[:, t_index]
        memory[tuple(indices)] = timestamps
        diff = -((i + 1) * dt + start_t - memory)
        surf = np.exp(diff / tau)
        all_surfaces.append(surf)
    return np.array(all_surfaces)
