import os
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Optional, Tuple, Union

import numpy as np
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    IterDataPipe,
    Mapper,
    Saver,
    ZipArchiveLoader,
)

from tonic.download_utils import download_url
from tonic.io import read_mnist_file

from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256


class NMNISTFileReader(IterDataPipe[Sample]):
    def __init__(
        self,
        dp: Union[IterDataPipe[str], IterDataPipe[Tuple[str, BinaryIO]]],
        dtype: Optional[np.dtype] = np.dtype(
            [("x", int), ("y", int), ("t", int), ("p", int)]
        ),
        keep_compressed: Optional[bool] = False,
    ) -> None:
        self.dp = dp
        self.dtype = dtype
        self.keep_cmp = keep_compressed

    def __iter__(self) -> Iterator[Sample]:
        if self.keep_cmp:
            for fname, fdata in self.dp:
                yield (
                    read_mnist_file(fdata, self.dtype, is_stream=True),
                    self._get_target(fname),
                )
        else:
            for fname in self.dp:
                yield (
                    read_mnist_file(fname, self.dtype),
                    self._get_target(fname),
                )

    def _get_target(self, fname: str) -> int:
        return int(fname.split(os.sep)[-2])


class NMNIST(Dataset):
    """`N-MNIST <https://www.garrickorchard.com/datasets/n-mnist>`_

    Events have (xytp) ordering.
    ::

        @article{orchard2015converting,
          title={Converting static image datasets to spiking neuromorphic datasets using saccades},
          author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={437},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        root (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        first_saccade_only (bool): If True, only work with events of the first of three saccades.
                                   Results in about a third of the events overall.
    """

    _DTYPE = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    _BASE_URL = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
    _TRAIN_URL = _BASE_URL + "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded"
    _TRAIN_FILENAME = "train.zip"
    _TRAIN_SHA256 = "1a54ee392a5e5082a0bef52911cd9211f63b950a4905ccd8890553804d3335f9"
    _TRAIN_FOLDER = "Train"
    _TEST_URL = _BASE_URL + "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded"
    _TEST_FILENAME = "test.zip"
    _TEST_SHA256 = "6ecfd5d85dbb49a631961d8dc3101871c5be53e645004ee34064f6557d169f09"
    _TEST_FOLDER = "Test"
    sensor_size = (34, 34, 2)

    def __init__(
        self,
        root: os.PathLike,
        train: Optional[bool] = True,
        first_saccade_only: Optional[bool] = False,
        keep_compressed: Optional[bool] = False,
    ) -> None:
        self.train = train
        self.first_saccade_only = first_saccade_only
        self.filename = self._TRAIN_FILENAME if self.train else self._TEST_FILENAME
        self.subfolder = self._TRAIN_FOLDER if self.train else self._TEST_FOLDER
        sha256 = self._TRAIN_SHA256 if self.train else self._TEST_SHA256
        root = Path(root, self.__class__.__name__)
        if not self._check_exists(root / self.filename):
            self._download(root, self.filename, sha256)
        super().__init__(
            root=root,
            keep_compressed=keep_compressed,
        )

    def __len__(self) -> int:
        return 60_000 if self.train else 10_000

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".bin")

    def _saccade_filter(self, events: np.ndarray):
        return events[events["t"] <= int(1e5)]

    def _download(self, root, filename, sha256) -> None:
        url = self._TRAIN_URL if self.train else self._TEST_URL
        download_url(url=url, root=root, filename=filename)
        check_sha256(root / filename, sha256)

    def _check_exists(self, folder) -> bool:
        return not folder.is_file()

    def _is_unzipped(self, folder) -> bool:
        if not folder.is_dir():
            return False
        dp = FileLister(str(folder), recursive=True).filter(self._filter)
        return len(list(dp)) >= 60_000 if self.train else 10_000

    def _datapipe(self) -> IterDataPipe[Sample]:
        if not self.keep_cmp:
            folder = self._root / self.subfolder
            if not self._is_unzipped(folder):
                with zipfile.ZipFile(self._root / self.filename, "r") as zip_file:
                    zip_file.extractall(self._root)
            dp = FileLister(str(folder), recursive=True)
            dp = dp.filter(self._filter)
        else:
            zip_filepath = self._root / self.filename
            dp = FileLister(str(zip_filepath))
            dp = FileOpener(dp, mode="rb")
            dp = ZipArchiveLoader(dp)
        # Reading data to structured NumPy array and integer target.
        dp = NMNISTFileReader(dp, keep_compressed=self.keep_cmp)
        # Filtering the first saccade.
        if self.first_saccade_only:
            dp = Mapper(dp, self._saccade_filter, input_col=0)
        return dp
