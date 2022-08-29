from .utils._dataset import Dataset, Sample
from typing import Optional, Union, Tuple, Iterator, Any, BinaryIO, Callable
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


class STMNISTFileReader(IterDataPipe[Sample]):
    def __init__(
        self,
        dp: IterDataPipe[Tuple[str, BinaryIO]],
        sensor_size: Optional[Tuple[int, int, int]] = (10, 10, 2),
        dtype: Optional[np.dtype] = np.dtype(
            [("x", int), ("y", int), ("t", int), ("p", int)]
        ),
    ) -> None:
        self.dp = dp
        self.dtype = dtype
        self.sensor_size = sensor_size

    def __iter__(self) -> Iterator[Sample]:
        for fname, fdata in self.dp:
            yield (
                self._mat_to_array(fdata),
                self._get_target(fname),
            )

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
    """
    Novel neuromorphic Spiking Tactile MNIST (ST-MNIST) dataset, which comprises handwritten digits obtained by human participants writing on a neuromorphic tactile sensor array.
    The original paper can be found at https://arxiv.org/abs/2005.04319.
    Data is provided with the MAT format. 
    Download of the compressed dataset has to be done by the user by accessing https://scholarbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zip, where a form has to be compiled. Then, the path to the ZIP archive has to be provided to the stmnist() function root argument.

    Events have (xytp) ordering.
    Parameters:
        root (string): path to the ZIP archive downloaded by the user (e.g. "./STMNIST.zip").
        transform (callable, optional): a callable of transforms to be applied to events data.
        target_transform (callable, optional): a callable of transform to be applied to target data.
        transforms (callable, optional): a callable of transforms to be applied to both events and target data.

    Returns:
        dp (IterDataPipe[Sample]): Torchdata data pipe that yields a tuple of events (or transformed events) and target.
    """


    _DTYPE = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    sensor_size = (10, 10, 2)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform, target_transform, transforms)

    def __len__(self) -> int:
        return 6_953

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".mat") and ("LUT" not in fname)

    def _datapipe(self) -> IterDataPipe[Sample]:
        dp = FileLister(str(self._root))
        dp = FileOpener(dp, mode="b")
        # Unzipping.
        dp = ZipArchiveLoader(dp)
        # Filtering the LUT and non-MAT files.
        dp = Filter(dp, self._filter, input_col=0)
        # Reading data to structured NumPy array and integer target.
        dp = STMNISTFileReader(dp)
        # Applying transforms.
        if self.transforms:
            dp = Mapper(dp, self.transforms)
        else:
            if self.transform:
                dp = Mapper(dp, self.transform, input_col=0, output_col=0)
            if self.target_transform:
                dp = Mapper(
                    dp, self.target_transform, input_col=1, output_col=1
                )
        return dp
