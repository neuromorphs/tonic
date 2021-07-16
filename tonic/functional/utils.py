import warnings
import numpy as np


# slicing functions adapted from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/preprocess.py
def slice_by_time(
    events: np.ndarray,
    ordering: str,
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
    assert "t" in ordering
    t_index = ordering.find("t")
    times = events[:, t_index]
    stride = time_window - overlap

    if include_incomplete:
        n_slices = int(np.ceil(((times[-1] - times[0]) - time_window) / stride) + 1)
    else:
        n_slices = int(np.floor(((times[-1] - times[0]) - time_window) / stride) + 1)

    window_start_times = np.arange(n_slices) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)
    return [events[indices_start[i] : indices_end[i], :] for i in range(n_slices)]


def slice_by_time_bins(
    events: np.ndarray, ordering: str, bin_count: int, overlap: float = 0.0
):
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
    assert "t" in ordering
    assert overlap < 1
    t_index = ordering.find("t")
    times = events[:, t_index]
    time_window = times[-1] // bin_count * (1 + overlap)
    stride = time_window * (1 - overlap)

    window_start_times = np.arange(bin_count) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)
    return [events[indices_start[i] : indices_end[i], :] for i in range(bin_count)]


def slice_by_spike_count(
    events: np.ndarray,
    ordering: str,
    spike_count: int,
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
    n_events = len(events)
    stride = spike_count - overlap

    if include_incomplete:
        n_slices = int(np.ceil((n_events - spike_count) / stride) + 1)
    else:
        n_slices = int(np.floor((n_events - spike_count) / stride) + 1)

    indices_start = (np.arange(n_slices) * stride).astype(int)
    indices_end = indices_start + spike_count
    return [events[indices_start[i] : indices_end[i], :] for i in range(n_slices)]


def slice_by_event_bins(
    events: np.ndarray, ordering: str, bin_count: int, overlap: float = 0.0
):
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
    return [events[indices_start[i] : indices_end[i], :] for i in range(bin_count)]


def is_multi_image(images, sensor_size):
    """
    Guesses at if there are multiple images inside of images

    Arguments:
    - images - image array to find where sensor_size is supported shapes
               include
               - [num_images, height, width, num_channels]
               - [height, width, num_channels]
               - [num_images, height, width]
               - [height, width]
    - sensor_size - sensor [W,H]

    Returns:
    - guess - best guess at if there are multiple images
    """

    warnings.warn("[Tonic]::Guessing if there are multiple images")
    if len(images.shape) == 4:
        guess = True
    elif len(images.shape) == 3:
        if images.shape[0] == sensor_size[0]:
            guess = False  # HWC
        else:
            guess = True  # NHW
    elif len(images.shape) == 2:
        guess = False
    else:
        raise NotImplementedError()
    warnings.warn("[Tonic]::Guessed [%s]" % str(guess))

    return guess
