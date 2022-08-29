from tonic.prototype.datasets._dataset import Dataset
from typing import Optional, Union, Tuple, Iterator, Any, BinaryIO, TypedDict, Callable
import numpy as np
import pathlib
from torchdata.datapipes.iter import (
    IterDataPipe,
    Zipper,
    ZipArchiveLoader,
    FileOpener,
    Filter,
    FileLister,
    Mapper,
)
from scipy.io import loadmat


class EventSample(TypedDict):
    events: np.ndarray
    target: str


class STMNISTFileReader(IterDataPipe[EventSample]):
    def __init__(
        self,
        dp: IterDataPipe[Tuple[Any, BinaryIO]],
        sensor_size: Optional[Tuple[int, int, int]] = (10, 10, 2),
        dtype: Optional[np.dtype] = np.dtype(
            [("x", int), ("y", int), ("t", int), ("p", int)]
        ),
    ) -> None:
        self.dp = dp
        self.dtype = dtype
        self.sensor_size = sensor_size

    def __iter__(self) -> Iterator[EventSample]:
        for fname, fdata in self.dp:
            yield {
                "events": self._mat_to_array(fdata),
                "target": self._get_target(fname),
            }

    def _get_target(self, fname: str) -> int:
        return int(fname.split("/")[-2])

    def _mat_to_array(self, f):
        # Transposing since the order is (address, event),
        # but we like (event, address).
        mat = loadmat(f)
        spiketrain = mat["spiketrain"].T
        # Separating coordinates and timestamps.
        spikes, timestamps = spiketrain[:, :-1], spiketrain[:, -1]
        # Getting events addresses.
        # First entry -> Event number.
        # Second entry -> Event address in [0,100).
        events_nums, events_addrs = spikes.nonzero()
        # Mapping addresses to 2D coordinates.
        # The mapping is (x%address, y//address), from the paper.
        events = np.zeros((len(events_nums)), dtype=self.dtype)
        events["x"] = events_addrs % self.sensor_size[0]
        events["y"] = events_addrs // self.sensor_size[1]
        # Converting floating point seconds to integer microseconds.
        events["t"] = (timestamps[events_nums] * 1e6).astype(int)
        # Converting -1 polarities to 0.
        events["p"] = np.max(spikes[(events_nums, events_addrs)], 0).astype(int)
        return events


class STMNIST(Dataset):

    _DTYPE = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    sensor_size = (10, 10, 2)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.transforms = transforms
        self.target_transform = target_transform
        self.transform = transform
        super().__init__(root)

    def __len__(self) -> int:
        return 6_953

    def _filter(self, dp: IterDataPipe[Tuple[str, BinaryIO]]) -> bool:
        return dp[0].endswith(".mat") and ("LUT" not in dp[0])

    def _datapipe(self) -> IterDataPipe[EventSample]:
        dp = FileLister(str(self._root))
        dp = FileOpener(dp, mode="b")
        # Unzipping.
        dp = ZipArchiveLoader(dp)
        # Filtering the LUT and non-MAT files.
        dp = Filter(dp, self._filter)
        # Reading data to structured NumPy array and integer target.
        dp = STMNISTFileReader(dp)
        # Applying transforms.
        if self.transforms:
            # The datapipe contains a dictionary. This can cause some trouble.
            dp = Mapper(dp, self.transforms)
        else:
            if self.transform:
                dp = Mapper(dp, self.transform, input_col="events", output_col="events")
            if self.target_transform:
                dp = Mapper(
                    dp, self.target_transform, input_col="target", output_col="target"
                )
        return dp
