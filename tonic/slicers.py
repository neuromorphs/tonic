from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from typing_extensions import Protocol


class Slicer(Protocol):
    """Base protocol class for slicers in Tonic.

    That means that you don't have to directly inherit from it, but just implement its methods.
    """

    def get_slice_metadata(self, data: Any, targets: Any) -> List[Tuple[Any]]:
        """This method returns the metadata for each recording that helps with slicing, for example
        the indices or timestamps at which the data would be sliced. The return value is typically
        a list of tuples that contain start and stop information for each slice.

        Parameters:
            data: Normally a tuple of data pieces.
            target: Normally a tuple of target pieces.

        Returns:
            metadata as a list of tuples of start and end indices, timestamps, etc.
        """
        ...

    def slice_with_metadata(self, data: Any, targets: Any, metadata: Any) -> List[Any]:
        """Given a piece of data and/or targets, cut out a certain part of it based on the
        start/end information given in metadata.

        Parameters:
            data: Normally a tuple of data pieces.
            target: Normally a tuple of target pieces.
            metadata: An array that contains start and stop information about one slice.

        Returns:
            A subset of the original data/targets which is a slice.
        """
        ...

    def slice(self, data: Any, targets: Any) -> List[Any]:
        """Generate metadata and return all slices at once.

        Parameters:
            data: Normally a tuple of data pieces.
            target: Normally a tuple of target pieces.

        Returns:
            The whole data and targets sliced into smaller slices.
        """
        ...


@dataclass(frozen=True)
class SliceByTime:
    """Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    time_window: float
    overlap: float = 0.0
    include_incomplete: bool = False

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


@dataclass
class SliceByTimeBins:
    """
    Slices data and targets along fixed number of bins of time length time_duration / bin_count * (1 + overlap).
    This method is good if your recordings all have roughly the same time length and you want an equal
    number of bins for each recording. Targets are copied.

    Parameters:
        bin_count (int): number of bins
        overlap (float): overlap specified as a proportion of a bin, needs to be smaller than 1. An overlap of 0.1
                    signifies that the bin will be enlarged by 10%. Amount of bins stays the same.
    """

    bin_count: int
    overlap: float = 0

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        events = data
        assert "t" in events.dtype.names
        assert self.overlap < 1

        times = events["t"]
        time_window = (times[-1] - times[0]) // self.bin_count * (1 + self.overlap)
        stride = time_window * (1 - self.overlap)

        window_start_times = np.arange(self.bin_count) * stride + times[0]
        window_end_times = window_start_times + time_window
        indices_start = np.searchsorted(times, window_start_times)
        indices_end = np.searchsorted(times, window_end_times)
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


@dataclass(frozen=True)
class SliceByEventCount:
    """Slices data and targets along a fixed number of events and overlap size. The number of bins
    depends on the amount of events in the recording. Targets are copied.

    Parameters:
        event_count (int): number of events for each bin
        overlap (int): overlap in number of events
        include_incomplete (bool): include the last incomplete slice that has fewer events
    """

    event_count: int
    overlap: int = 0
    include_incomplete: bool = False

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        n_events = len(data)
        event_count = min(self.event_count, n_events)

        stride = self.event_count - self.overlap
        if stride <= 0:
            raise Exception("Inferred stride <= 0")

        if self.include_incomplete:
            n_slices = int(np.ceil((n_events - event_count) / stride) + 1)
        else:
            n_slices = int(np.floor((n_events - event_count) / stride) + 1)

        indices_start = (np.arange(n_slices) * stride).astype(int)
        indices_end = indices_start + event_count
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


@dataclass(frozen=True)
class SliceByEventBins:
    """
    Slices an event array along fixed number of bins that each have n_events // bin_count * (1 + overlap) events.
    This slicing method is good if you recordings have all roughly the same amount of overall activity in the scene
    and you want an equal number of bins for each recording. Targets are copied.

    Parameters:
        bin_count (int): number of bins
        overlap (float): overlap in proportion of a bin, needs to be smaller than 1. An overlap of 0.1
                    signifies that the bin will be enlarged by 10%. Amount of bins stays the same.
    """

    bin_count: int
    overlap: float = 0

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        n_events = len(data)
        spike_count = int(n_events // self.bin_count * (1 + self.overlap))
        stride = int(spike_count * (1 - self.overlap))

        indices_start = np.arange(self.bin_count) * stride
        indices_end = indices_start + spike_count
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


@dataclass
class SliceAtIndices:
    """Slices data at the specified event indices. Targets are copied.

    Parameters:
        start_indices (list): List of start indices
        end_indices (list): List of end indices (exclusive)
    """

    start_indices: np.ndarray
    end_indices: np.ndarray

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        return list(zip(self.start_indices, self.end_indices))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


@dataclass
class SliceAtTimePoints:
    """Slice the data at the specified time points.

    Parameters:
        tw_start (list): List of start times
        tw_end (list): List of end times
    """

    start_tw: np.ndarray
    end_tw: np.ndarray

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        indices_start = np.searchsorted(t, self.start_tw)
        indices_end = np.searchsorted(t, self.end_tw)
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[int, int]]
    ):
        return [data[start:end] for start, end in metadata], targets


def slice_events_by_time(
    events: np.ndarray,
    time_window: int,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    return SliceByTime(
        time_window=time_window, overlap=overlap, include_incomplete=include_incomplete
    ).slice(events, None)[0]


def slice_events_by_time_bins(events: np.ndarray, bin_count: int, overlap: float = 0.0):
    return SliceByTimeBins(bin_count=bin_count, overlap=overlap).slice(events, None)[0]


def slice_events_by_count(
    events: np.ndarray,
    event_count: int,
    overlap: int = 0,
    include_incomplete: bool = False,
):
    return SliceByEventCount(
        event_count=event_count, overlap=overlap, include_incomplete=include_incomplete
    ).slice(events, None)[0]


def slice_events_by_event_bins(
    events: np.ndarray, bin_count: int, overlap: float = 0.0
):
    return SliceByEventBins(bin_count=bin_count, overlap=overlap).slice(events, None)[0]


def slice_events_at_indices(events: np.ndarray, start_indices, end_indices):
    return SliceAtIndices(start_indices=start_indices, end_indices=end_indices).slice(
        events, None
    )[0]


def slice_events_at_timepoints(
    events: np.ndarray, start_tw: np.ndarray, end_tw: np.ndarray
) -> List[np.ndarray]:
    return SliceAtTimePoints(start_tw=start_tw, end_tw=end_tw).slice(events, None)[0]
