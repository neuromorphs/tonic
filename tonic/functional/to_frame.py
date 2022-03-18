import numpy as np
import math
from .slicing import (
    slice_by_time,
    slice_by_event_count,
    slice_by_time_bins,
    slice_by_event_bins,
)


def to_frame_numpy(
    events,
    sensor_size,
    time_window=None,
    event_count=None,
    n_time_bins=None,
    n_event_bins=None,
    overlap=0.0,
    include_incomplete=False,
):
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (event_count) or constant number of frames (n_time_bins / n_event_bins).

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H,P]
        time_window (None): window length in us.
        event_count (None): number of events per frame.
        n_time_bins (None): fixed number of frames, sliced along time axis.
        n_event_bins (None): fixed number of frames, sliced along number of events in the recording.
        overlap (0.): overlap between frames defined either in time in us, number of events or number of bins.
        include_incomplete (False): if True, includes overhang slice when time_window or event_count is specified. Not valid for bin_count methods.

    Returns:
        numpy array with dimensions (TxPxHxW)
    """
    assert "x" and "t" and "p" in events.dtype.names

    if (
        not sum(
            param is not None
            for param in [time_window, event_count, n_time_bins, n_event_bins]
        )
        == 1
    ):
        raise ValueError(
            "Please assign a value to exactly one of the parameters time_window,"
            " event_count, n_time_bins or n_event_bins."
        )

    if not sensor_size:
        sensor_size_x = int(events["x"].max() + 1)
        sensor_size_p = len(np.unique(events["p"]))
        if "y" in events.dtype.names:
            sensor_size_y = int(events["y"].max() + 1)
            sensor_size = (sensor_size_x, sensor_size_y, sensor_size_p)
        else:
            sensor_size = (sensor_size_x, 1, sensor_size_p)

    # test for single polarity
    if sensor_size[2] == 1:
        events["p"] = 0

    if time_window:
        event_slices = slice_by_time(
            events, time_window, overlap=overlap, include_incomplete=include_incomplete
        )
    elif event_count:
        event_slices = slice_by_event_count(
            events, event_count, overlap=overlap, include_incomplete=include_incomplete
        )
    elif n_time_bins:
        event_slices = slice_by_time_bins(events, n_time_bins, overlap=overlap)
    elif n_event_bins:
        event_slices = slice_by_event_bins(events, n_event_bins, overlap=overlap)

    if "y" in events.dtype.names:
        frames = np.zeros((len(event_slices), *sensor_size[::-1]), dtype=np.int16)
        for i, event_slice in enumerate(event_slices):
            np.add.at(
                frames,
                (i, event_slice["p"].astype(int), event_slice["y"], event_slice["x"]),
                1,
            )
    else:
        frames = np.zeros((len(event_slices), sensor_size[2], sensor_size[0]), dtype=np.int16)
        for i, event_slice in enumerate(event_slices):
            np.add.at(frames, (i, event_slice["p"].astype(int), event_slice["x"]), 1)
    return frames
