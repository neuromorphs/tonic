from dataclasses import dataclass
from typing_extensions import Protocol
from typing import Any, List, Tuple
from . import functional
import numpy as np

# some slicing methods have been copied and/or adapted from
# https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/preprocess.py
class Slicer(Protocol):
    def get_slice_metadata(self, data: Any) -> List[Any]:
        """
        This method returns the meta data that helps with slicing of the given data.
        Eg. it could return the indices at which the data would be sliced.
        The return value of this method should be usable by the method slice_with_meta.

        Parameters:
            data:

        Returns:
            metadata
        """
        ...

    def slice_with_metadata(self, data: Any, metadata: Any) -> List[Any]:
        """
        Slice the data using the metadata.
        Parameters:
            data:
            metadata:

        Returns:
            sliced_data
        """
        ...

    def slice(self, data: Any) -> List[Any]:
        """
        Slice the given data and return the sliced data
        Parameters:
            data:

        Returns:
            sliced_data
        """
        ...


@dataclass(frozen=True)
class SliceByTime:
    """
    Return xytp split according to fixed timewindow and overlap size
    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window: int
            Length of time for each xytp (ms)
        overlap: int
            Length of time of overlapping (ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp
    """

    time_window: float
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

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]
        return list(zip(indices_start, indices_end))

    @staticmethod
    def slice_with_metadata(data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass(frozen=True)
class SliceByEventCount:
    """
    Return xytp sliced to equal number of events specified by event_count

    Parameters:
        event_count (int):  Number of events per xytp
        overlap: int
            No. of spikes overlapping in the following xytp(ms)
        include_incomplete: bool
            include incomplete slices ie potentially the last xytp
    """

    event_count: int
    overlap: int = 0
    include_incomplete: bool = False

    def slice(self, data: np.ndarray) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data)
        return self.slice_with_metadata(data, metadata)

    def get_slice_metadata(self, data: np.ndarray) -> List[Tuple[int, int]]:
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
    def slice_with_metadata(data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass
class SliceAtIndices:
    """
    Slices data at the specified indices

    Parameters:
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

    @staticmethod
    def slice_with_metadata(data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]


@dataclass
class SliceAtTimePoints:
    """
    Slice the data at the specified time points

    Parameters:
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

    @staticmethod
    def slice_with_metadata(data: np.ndarray, metadata: List[Tuple[int, int]]):
        return [data[start:end] for start, end in metadata]
