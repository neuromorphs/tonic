import numpy as np
import math
from .utils import slice_by_time, slice_by_spike_count, slice_by_time_bins, slice_by_event_bins


# adapted in parts from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/dataset_generator.py
def to_frame_numpy(
    events, sensor_size, ordering, time_window=None, spike_count=None, n_time_bins=None, n_event_bins=None, overlap=0., include_incomplete=False, merge_polarities=False
):
    """Accumulate events to frames by slicing along constant time (time_window), 
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events.
        time_window (None): window length in us.
        spike_count (None): number of events per frame.
        n_time_bins (None): fixed number of frames, sliced along time axis.
        n_event_bins (None): fixed number of frames, sliced along number of events in the recording.
        overlap (0.): overlap between frames defined either in time in us, number of events or number of bins.
        include_incomplete (False): if True, includes overhang slice when time_window or spike_count is specified. Not valid for bin_count methods.
        merge_polarities (False): if True, merge polarity channels to a single channel.

    Returns:
        numpy array of n rate-coded frames with channels p: (NxPxWxH)
    """
    assert "x" and "y" and "t" and "p" in ordering
    assert len(sensor_size) == 2
    if not sum(param is not None for param in [time_window, spike_count, n_time_bins, n_event_bins]) == 1:
        raise ValueError("Please assign a value to exactly one of the parameters time_window, spike_count, n_time_bins or n_event_bins.")
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_events = len(events)

    pols = events[:, p_index]
    if merge_polarities:
        pols[pols == -1] = 1
        pols[pols == 0] = 1
    else:
        pols[pols == -1] = 0

    if time_window:
        event_slices = slice_by_time(events, ordering, time_window, overlap=overlap, include_incomplete=include_incomplete)
    elif spike_count:
        event_slices = slice_by_spike_count(events, ordering, spike_count, overlap=overlap, include_incomplete=include_incomplete)
    elif n_time_bins:
        event_slices = slice_by_time_bins(events, ordering, n_time_bins, overlap=overlap)
    elif n_event_bins:
        event_slices = slice_by_event_bins(events, ordering, n_event_bins, overlap=overlap)
    
    bins_p = len(np.unique(pols))
    bins_y, bins_x = (range(sensor_size[0] + 1), range(sensor_size[1] + 1))
    
    frames = np.empty((len(event_slices), bins_p, len(bins_y) - 1, len(bins_x) - 1), dtype=np.uint16)
    for i, event_slice in enumerate(event_slices):
        frames[i] = np.histogramdd((event_slice[:,p_index], event_slice[:,y_index], event_slice[:,x_index]),
                                   bins=(np.arange(bins_p+1), bins_y, bins_x))[0]
    return frames
