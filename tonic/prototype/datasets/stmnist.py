import os
import pathlib
from typing import Any, BinaryIO, Callable, Iterator, Optional, Tuple, Union

import numpy as np
from scipy.io import loadmat
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    IterDataPipe,
    Mapper,
    Saver,
    ZipArchiveLoader,
    Zipper,
)

from tonic.download_utils import check_integrity

from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256


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
        return int(fname.split(os.sep)[-2])

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
        events["p"] = np.maximum(spikes[(events_nums, events_addrs)], 0).astype(int)
        return events


class STMNIST(Dataset):
    """Novel neuromorphic Spiking Tactile MNIST (ST-MNIST) dataset, which comprises handwritten
    digits obtained by human participants writing on a neuromorphic tactile sensor array. The
    original paper can be found at https://arxiv.org/abs/2005.04319. Data is provided with the MAT
    format. Download of the compressed dataset has to be done by the user by accessing https://scho
    larbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zi
    p, where a form has to be compiled. Then, the path to the ZIP archive has to be provided to the
    stmnist() function root argument.

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
    _SHA256 = "825bb5a64753fff4a2a2c32e3497fa8a951d9c94993e03ba25a057e17d83b884"
    sensor_size = (10, 10, 2)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        keep_compressed: Optional[bool] = False,
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        super().__init__(
            root,
            transform,
            target_transform,
            transforms,
            keep_compressed,
            skip_sha256_check,
        )
        assert self._check_exists(), "Error: the archive is not present."
        if not self.skip_sha256:
            check_sha256(fpath=self._root, sha256_provided=self._SHA256)

    def _check_exists(self):
        return os.path.isfile(self._root)

    def __len__(self) -> int:
        return 6_953

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".mat") and ("LUT" not in fname)

    def _uncompress(
        self, dp: IterDataPipe[Tuple[Any, BinaryIO]]
    ) -> IterDataPipe[Tuple[str, BinaryIO]]:
        # Stripping the archive from self._root.
        root = os.sep.join(str(self._root).split(os.sep)[:-1])
        # Joining root with a folder to contain the data.
        root = os.path.join(root, "data_uncompressed")
        if not os.path.isdir(root):
            os.mkdir(root)
            # Decompressing in root.
            def read_bin(fdata):
                return fdata.read()

            def filepath_fn(fpath):
                fpath_i = fpath.split(os.sep)
                start = fpath_i.index("data_submission") + len(os.sep)
                fpath_i = os.sep.join(fpath_i[start:])
                return os.path.join(
                    root,
                    fpath_i,
                )

            dp = Mapper(dp, read_bin, input_col=1)
            dp = Saver(dp, mode="wb", filepath_fn=filepath_fn)
            # Saving data to file.
            for x in dp:
                pass
        dp = FileLister(root, recursive=True)
        dp = FileOpener(dp, mode="rb")
        return dp

    def _datapipe(self) -> IterDataPipe[Sample]:
        dp = FileLister(str(self._root))
        dp = FileOpener(dp, mode="b")
        # Unzipping.
        dp = ZipArchiveLoader(dp)
        if not self.keep_cmp:
            dp = self._uncompress(dp)
        dp = Filter(dp, self._filter, input_col=0)
        dp = STMNISTFileReader(dp)
        return dp
