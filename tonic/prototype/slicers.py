from dataclasses import dataclass

import numpy as np
import torchdata.datapipes.iter as pipes
from torchdata.datapipes import functional_datapipe


@dataclass
@functional_datapipe("slice_by_time")
class SliceByTime(pipes.IterDataPipe):
    """Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Only works on numpy event arrays that contain a 't' or 'ts'
    field.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    source_dp: pipes.IterDataPipe
    dt: float
    overlap: float = 0.0
    include_incomplete: bool = False

    def __iter__(self):
        it = iter(self.source_dp)
        while True:
            try:
                events = next(it)
                if "t" in events.dtype.names:
                    t = events["t"]
                elif "ts" in events.dtype.names:
                    t = events["ts"]
                stride = self.dt - self.overlap
                assert stride > 0
                rounding_fn = np.ceil if self.include_incomplete else np.floor
                n_slices = int(rounding_fn(((t[-1] - t[0]) - self.dt) / stride) + 1)
                n_slices = max(n_slices, 1)  # for strides larger than recording time

                window_start_times = np.arange(n_slices) * stride + t[0]
                window_end_times = window_start_times + self.dt
                indices_start = np.searchsorted(t, window_start_times)[:n_slices]
                indices_end = np.searchsorted(t, window_end_times)[:n_slices]
                for start, end in zip(indices_start, indices_end):
                    yield events[start:end]
            except StopIteration:
                return


@dataclass
@functional_datapipe("slice_by_event_count")
class SliceByEventCount(pipes.IterDataPipe):
    """Slices data and targets along a fixed number of events and overlap size. The number of bins
    depends on the amount of events in the recording. Only works on numpy event arrays.

    Parameters:
        event_count (int): number of events for each bin
        overlap (int): overlap in number of events
        include_incomplete (bool): include the last incomplete slice that has fewer events
    """

    source_dp: pipes.IterDataPipe
    n: int
    overlap: int = 0
    include_incomplete: bool = False

    def __iter__(self):
        it = iter(self.source_dp)
        while True:
            try:
                events = next(it)
                n_events = len(events)
                event_count = min(self.n, n_events)
                stride = self.n - self.overlap
                if stride <= 0:
                    raise Exception(
                        "Inferred stride <= 0. Increase n or decrease overlap."
                    )
                rounding_fn = np.ceil if self.include_incomplete else np.floor
                n_slices = int(rounding_fn((n_events - event_count) / stride) + 1)
                for start in (np.arange(n_slices) * stride).astype(int):
                    yield events[start : start + event_count]
            except StopIteration:
                return
