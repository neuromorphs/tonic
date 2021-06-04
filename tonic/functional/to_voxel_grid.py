import numpy as np
from .utils import is_multi_image


def to_voxel_grid_numpy(events, sensor_size, ordering, num_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Code taken from https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L431

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        ordering: ordering of the event tuple inside of events. This function requires 'x', 'y',
                  't' and 'p' to be in the ordering.
        num_time_bins: number of bins in the temporal axis of the voxel grid.

    Returns:
        numpy array of n event volumes (n,w,h,t)

    """
    assert "x" in ordering and "y" in ordering
    assert "t" in ordering and "p" in ordering

    x_loc = ordering.index("x")
    y_loc = ordering.index("y")
    t_loc = ordering.index("t")
    p_loc = ordering.index("p")

    voxel_grid = np.zeros(
        (num_time_bins, sensor_size[1], sensor_size[0]), np.float32
    ).ravel()

    # normalize the event timestamps so that they lie between 0 and num_time_bins
    last_stamp = events[-1, t_loc]
    first_stamp = events[0, t_loc]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, t_loc] = (num_time_bins - 1) * (events[:, t_loc] - first_stamp) / deltaT
    ts = events[:, t_loc]
    xs = events[:, x_loc].astype(np.int)
    ys = events[:, y_loc].astype(np.int)
    pols = events[:, p_loc]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < num_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(voxel_grid, (num_time_bins, sensor_size[1], sensor_size[0]))

    return voxel_grid
