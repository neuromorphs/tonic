import os
import pathlib
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
    Zipper,
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
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
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
        root: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        train: Optional[bool] = True,
        first_saccade_only: Optional[bool] = False,
        keep_compressed: Optional[bool] = False,
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        self.train = train
        self.first_saccade_only = first_saccade_only
        super().__init__(
            root,
            transform,
            target_transform,
            transforms,
            keep_compressed,
            skip_sha256_check,
        )
        self._download()

    def __len__(self) -> int:
        return 60_000 if self.train else 10_000

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".bin")

    def _saccade_filter(self, events: np.ndarray):
        return events[events["t"] > int(1e5)]

    def _download(self) -> None:
        # Setting file path depending on train value.
        url = self._TRAIN_URL if self.train else self._TEST_URL
        sha256 = self._TRAIN_SHA256 if self.train else self._TEST_SHA256
        filename = self._TRAIN_FILENAME if self.train else self._TEST_FILENAME
        # Downloading and SHA256 check.
        if not self._check_exists():
            download_url(url=url, root=self._root, filename=filename)
            check_sha256(os.path.join(self._root, filename), sha256)
        elif not self.skip_sha256:
            check_sha256(os.path.join(self._root, filename), sha256)

    def _check_exists(self) -> bool:
        filename = self._TRAIN_FILENAME if self.train else self._TEST_FILENAME
        return os.path.isfile(os.path.join(self._root, filename))

    def _uncompress(
        self, dp: IterDataPipe[Tuple[Any, BinaryIO]]
    ) -> IterDataPipe[Tuple[str, BinaryIO]]:
        folder = self._TRAIN_FOLDER if self.train else self._TEST_FOLDER
        # Joining root with a folder to contain the data.
        filepath = os.path.join(self._root, folder)
        if (not os.path.isdir(filepath)) or (not os.listdir(filepath)):
            os.makedirs(filepath, exist_ok=True)
            # Decompressing in root.
            def read_bin(fdata):
                return fdata.read()

            dp = Mapper(dp, read_bin, input_col=1)

            def filepath_fn(fpath):
                fpath_i = fpath.split(os.sep)
                start = fpath_i.index(folder) + len(os.sep)
                fpath_i = os.sep.join(fpath_i[start:])
                return os.path.join(filepath, fpath_i)

            dp = Saver(dp, mode="wb", filepath_fn=filepath_fn)
            # Saving data to file.
            for x in dp:
                pass
        dp = FileLister(filepath, recursive=True)
        return dp

    def _datapipe(self) -> IterDataPipe[Sample]:
        filename = self._TRAIN_FILENAME if self.train else self._TEST_FILENAME
        filepath = os.path.join(self._root, filename)
        dp = FileLister(str(filepath))
        dp = FileOpener(dp, mode="b")
        # Unzipping.
        dp = ZipArchiveLoader(dp)
        if not self.keep_cmp:
            dp = self._uncompress(dp).filter(self._filter)
        else:
            # Filtering the non-bin files.
            dp = Filter(dp, self._filter, input_col=0)
        # Reading data to structured NumPy array and integer target.
        dp = NMNISTFileReader(dp, keep_compressed=self.keep_cmp)
        # Filtering the first saccade.
        if self.first_saccade_only:
            dp = Mapper(dp, self._saccade_filter, input_col=0)
        return dp
