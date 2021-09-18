from dataclasses import dataclass
from typing import Protocol, Any, List, Tuple

import numpy as np


class Slicer(Protocol):
    def get_slice_metadata(self, data: Any) -> List[Any]:
        """
        This method returns the meta data that helps with slicing of the given data.
        Eg. it could return the indices at which the data would be sliced.
        The return value of this method should be usable by the method slice_with_meta.

        Args:
            data:

        Returns:
            metadata
        """
        ...

    def slice_with_meta(self, data: Any, metadata: Any) -> List[Any]:
        """
        Slice the data using the metadata.
        Args:
            data:
            metadata:

        Returns:
            sliced_data
        """
        ...

    def slice(self, data: Any) -> List[Any]:
        """
        Slice the given data and return the sliced data
        Args:
            data:

        Returns:
            sliced_data
        """
        ...


@dataclass(frozen=True)
class SliceByTime:
    """
    Return xytp split according to fixed timewindow and overlap size
    <        <overlap>        >
    |   window1      |
             |   window2      |

    Args:
        time_window: int
            Length of time for each xytp (ms)
        overlap: int
            Length of time of overlapping (ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp
    """
    time_window: float = 1.0
    overlap: float = 0.0
    include_incomplete: bool = False

    def slice(self, data: np.ndarray) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data)
        return self.slice_with_metadata(data, metadata)

    def get_slice_metadata(self, data: np.ndarray) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        tw_start = np.arange(n_slices) * stride + t[0]
        tw_end = tw_start + self.time_window
        indices_start = np.searchsorted(t, tw_start)[:n_slices]
        indices_end = np.searchsorted(t, tw_end)[:n_slices]
        return list(zip(indices_start, indices_end))

    @classmethod
    def slice_with_metadata(cls, data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass(frozen=True)
class SliceByEventCount:
    """
    Return xytp sliced nto equal number of events specified by spike_count

    Args:
        spike_count (int):  Number of events per xytp
        overlap: int
            No. of spikes overlapping in the following xytp(ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp
    """
    spike_count: int
    overlap: int
    include_incomplete: bool

    def slice(self, data: np.ndarray) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data)
        return self.slice_with_metadata(data, metadata)

    def get_slice_metadata(self, data: np.ndarray) -> List[Tuple[int, int]]:
        n_spk = len(data)
        spike_count = min(self.spike_count, n_spk)
        stride = spike_count - self.overlap
        if stride <= 0:
            raise Exception("Inferred stride <= 0")

        if self.include_incomplete:
            n_slices = int(np.ceil((n_spk - spike_count) / stride) + 1)
        else:
            n_slices = int(np.floor((n_spk - spike_count) / stride) + 1)

        indices_start = np.arange(n_slices) * stride
        indices_end = indices_start + spike_count
        return list(zip(indices_start, indices_end))

    @classmethod
    def slice_with_metadata(cls, data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass
class SliceAtIndices:
    """
    Slices data at the specified indices

    Args:
        start_indices: (List[Int]): List of start indices
        end_indices: (List[Int]): List of end indices (exclusive)
    """
    start_indices: np.ndarray
    end_indices: np.ndarray

    def slice(self, data: np.ndarray) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data)
        return self.slice_with_metadata(data, metadata)

    def get_slice_metadata(self, _: np.ndarray) -> List[Tuple[int, int]]:
        return list(zip(self.start_indices, self.end_indices))

    @classmethod
    def slice_with_metadata(cls, data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass
class SliceAtTimePoints:
    """
    Slice the data at the specified time points

    Args:
        tw_start: (List[Int]): List of start times
        tw_end: (List[Int]): List of end times
    """
    start_tw: np.ndarray
    end_tw: np.ndarray

    def slice(self, data: np.ndarray) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data)
        return self.slice_with_metadata(data, metadata)

    def get_slice_metadata(self, data: np.ndarray) -> List[Tuple[int, int]]:
        t = data["t"]
        indices_start = np.searchsorted(t, self.start_tw)
        indices_end = np.searchsorted(t, self.end_tw)
        return list(zip(indices_start, indices_end))

    @classmethod
    def slice_with_metadata(cls, data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


########################### Functional ###########################


def slice_by_time(xytp: np.ndarray, time_window: int, overlap: int = 0, include_incomplete=False) -> List[np.ndarray]:
    """
    Return xytp split according to fixed timewindow and overlap size
    <        <overlap>        >
    |   window1      |
             |   window2      |

    Args:
        xytp: np.ndarray
            Structured array of events
        time_window: int
            Length of time for each xytp (ms)
        overlap: int
            Length of time of overlapping (ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp

    Returns:
        slices List[np.ndarray]: Data slices

    """
    slicer = SliceByTime(time_window=time_window, overlap=overlap, include_incomplete=include_incomplete)
    return slicer.slice(xytp)


def slice_by_count(xytp: np.ndarray, spike_count: int, overlap: int = 0, include_incomplete=False) -> List[np.ndarray]:
    """
    Return xytp sliced nto equal number of events specified by spike_count

    Args:
        xytp (np.ndarray):  Structured array of events
        spike_count (int):  Number of events per xytp
        overlap: int
            No. of spikes overlapping in the following xytp(ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp
    Returns:
        slices (List[np.ndarray]): Data slices
    """
    slicer = SliceByEventCount(spike_count=spike_count, overlap=overlap, include_incomplete=include_incomplete)
    return slicer.slice(xytp)


def slice_at_indices(xytp: np.ndarray, start_indices, end_indices):
    """
    Return xytp sliced at the specified indices

    Args:
    -----
        xytp (np.ndarray):  Structured array of events
        start_indices: (List[Int]): List of start indices
        end_indices: (List[Int]): List of end indices (exclusive)
    Returns:
    --------
    slices (np.ndarray): Data slices
    """
    slicer = SliceAtIndices(start_indices=start_indices, end_indices=end_indices)
    return slicer.slice(xytp)


def slice_at_time_points(xytp: np.ndarray, start_tw: np.ndarray, end_tw: np.ndarray) -> List[np.ndarray]:
    """
    Return xytp sliced at the specified time windows

    Args:
    -----
        xytp (np.ndarray):  Structured array of events
        start_tw: (np.ndarray): List of start time points
        end_tw: (np.ndarray): List of end time points
    Returns:
    --------
    slices (np.ndarray): Data slices
    """
    slicer = SliceAtTimePoints(start_tw=start_tw, end_tw=end_tw)
    return slicer.slice(xytp)
