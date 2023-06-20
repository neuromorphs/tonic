import os
import zipfile
from pathlib import Path
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
)

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
        for fname in self.dp:
            yield (
                self._mat_to_array(fname),
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
    """`ST-MNIST <https://arxiv.org/abs/2005.04319>`_

    Neuromorphic Spiking Tactile MNIST (ST-MNIST) dataset, which comprises handwritten
    digits obtained by human participants writing on a neuromorphic tactile sensor array. The
    original paper can be found at https://arxiv.org/abs/2005.04319. Data is provided with the MAT
    format. Download of the compressed dataset has to be done by the user by accessing https://scho
    larbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zi
    p, where a form has to be completed. Then, the path to the ZIP archive has to be provided to the
    STMNIST constructor root argument.

    Events have (xytp) ordering.
    Parameters:
        root (string): Parent folder of 'STMNIST/STMNIST dataset NUS Tee Research Group.zip'. The STMNIST folder is related to the Tonic class name and is needed currently.
        shuffle (bool): Whether to shuffle the dataset. More efficient if done based on file paths.

    Returns:
        dp (IterDataPipe[Sample]): Torchdata data pipe that yields a tuple of events (or transformed events) and target.
    """

    _DTYPE = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    _SHA256 = "825bb5a64753fff4a2a2c32e3497fa8a951d9c94993e03ba25a057e17d83b884"
    _FILENAME = "STMNIST dataset NUS Tee Research Group.zip"
    sensor_size = dict(x=10, y=10, p=2)

    def __init__(
        self,
        root: os.PathLike,
        keep_compressed: Optional[bool] = False,
        skip_sha256_check: Optional[bool] = True,
        shuffle: bool = False,
    ) -> None:
        super().__init__(
            Path(root, self.__class__.__name__),
            keep_compressed,
            skip_sha256_check,
        )
        if not skip_sha256_check:
            check_sha256(
                fpath=self._root / self._FILENAME, sha256_provided=self._SHA256
            )
        if not self._check_exists():
            assert os.path.isfile(
                self._root / self._FILENAME
            ), "Error: root must point to parent folder of STMNIST/STMNIST dataset NUS Tee Research Group.zip."
            if not keep_compressed:
                with zipfile.ZipFile(self._root / self._FILENAME, "r") as zip_file:
                    zip_file.extractall(self._root)
        self.shuffle = shuffle

    def _check_exists(self):
        dp = FileLister(str(self._root), recursive=True).filter(self._filter)
        return len(list(dp)) >= 6953

    def __len__(self) -> int:
        return 6_953

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".mat") and ("LUT" not in fname)

    def _datapipe(self) -> IterDataPipe[Sample]:
        dp = FileLister(str(self._root), recursive=True)
        if self.shuffle:
            dp = dp.shuffle(buffer_size=10_000)
        dp = Filter(dp, self._filter)
        dp = STMNISTFileReader(dp)
        return dp
