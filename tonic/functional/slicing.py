import warnings
import numpy as np
from tonic.slicers import (
    SliceByTime,
    SliceByEventCount,
    SliceAtIndices,
    SliceAtTimePoints,
)
from typing import List


def slice_by_time(
    events: np.ndarray,
    time_window: int,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    """
    Slices an event array along fixed time window and overlap size.
    The number of bins depends on the length of the recording.
             <overlap>
    |   window1      |
             |   window2      |

    Parameters:
        events: numpy array of events
        ordering (str): ordering of the events, i.e. "xytp"
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include incomplete slices

    Returns:
        list of event slices (np.ndarray)
    """
    assert "t" in events.dtype.names

    times = events["t"]
    stride = time_window - overlap

    if include_incomplete:
        n_slices = int(np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1)
    else:
        n_slices = int(np.floor(((times[-1] - times[0]) - time_window) / stride) + 1)

    window_start_times = np.arange(n_slices) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)
    return [events[indices_start[i] : indices_end[i]] for i in range(n_slices)]


def slice_by_time_bins(events: np.ndarray, bin_count: int, overlap: float = 0.0):
    """
    Slices an event array along fixed number of bins of time length max_time / bin_count * (1+overlap).
    This method is good if your recordings all have roughly the same time length and you want an equal
    number of bins for each recording.

    Parameters:
        events: numpy array of events
        ordering (str): ordering of the events, i.e. "xytp"
        bin_count (int): number of bins
        overlap (float): overlap in number of bins, needs to be smaller than 1. An overlap of 0.1
                    signifies that the bin will be enlarged by 10%. Amount of bins stays the same.

    Returns:
        list of event slices (np.ndarray)
    """
    assert "t" in events.dtype.names
    assert overlap < 1

    times = events["t"]
    time_window = (times[-1] - times[0]) // bin_count * (1 + overlap)
    stride = time_window * (1 - overlap)

    window_start_times = np.arange(bin_count) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)
    return [events[indices_start[i] : indices_end[i]] for i in range(bin_count)]


def slice_by_event_count(
    events: np.ndarray,
    event_count: int,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    """
    Slices an event array along fixed number of events and overlap size.
    The number of bins depends on the amount of events in the recording.

    Parameters:
        events: numpy array of events
        ordering (str): ordering of the events, i.e. "xytp"
        spike_count (int): number of events for each bin
        overlap (int): overlap in number of events
        include_incomplete (bool): include incomplete slices

    Returns:
        list of event slices (np.ndarray)
    """
    return SliceByEventCount(
        event_count=event_count, overlap=overlap, include_incomplete=include_incomplete
    ).slice(events)


def slice_by_event_bins(events: np.ndarray, bin_count: int, overlap: float = 0.0):
    """
    Slices an event array along fixed number of bins that each have n_events // bin_count * (1 + overlap) events.
    This slicing method is good if you recordings have all roughly the same amount of overall activity in the scene
    and you want an equal number of bins for each recording.

    Parameters:
        events: numpy array of events
        ordering (str): ordering of the events, i.e. "xytp"
        bin_count (int): number of bins
        overlap (float): overlap in number of bins, needs to be smaller than 1. An overlap of 0.1
                    signifies that the bin will be enlarged by 10%. Amount of bins stays the same.

    Returns:
        list of event slices (np.ndarray)
    """
    n_events = len(events)
    spike_count = int(n_events // bin_count * (1 + overlap))
    stride = int(spike_count * (1 - overlap))

    indices_start = np.arange(bin_count) * stride
    indices_end = indices_start + spike_count
    return [events[indices_start[i] : indices_end[i]] for i in range(bin_count)]


def slice_at_indices(xytp: np.ndarray, start_indices, end_indices):
    slicer = SliceAtIndices(start_indices=start_indices, end_indices=end_indices)
    return slicer.slice(xytp)


def slice_at_timepoints(
    xytp: np.ndarray, start_tw: np.ndarray, end_tw: np.ndarray
) -> List[np.ndarray]:
    slicer = SliceAtTimePoints(start_tw=start_tw, end_tw=end_tw)
    return slicer.slice(xytp)
