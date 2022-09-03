from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256
import os
from tonic.io import make_structured_array, read_mnist_file
from tonic.download_utils import download_url
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
    Saver,
)


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
                    self._bin_to_array(fdata),
                    self._get_target(fname),
                )
        else:
            for fname in self.dp:
                yield (
                    read_mnist_file(fname, self.dtype),
                    self._get_target(fname),
                )

    def _get_target(self, fname: str) -> int:
        return int(fname.split("/")[-2])

    def _bin_to_array(self, bin_stream: BinaryIO):
        """
        Reads the events contained in N-MNIST/N-CALTECH101 datasets.
        Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
        """
        raw_data = np.frombuffer(bin_stream.read(), dtype=np.uint8, offset=0).astype(
            np.uint32
        )

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = (
            ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        )

        # Process time stamp overflow events
        time_increment = 2**13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        xytp = make_structured_array(
            all_x[td_indices],
            all_y[td_indices],
            all_ts[td_indices],
            all_p[td_indices],
            dtype=self.dtype,
        )
        return xytp


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
    ) -> None:
        self.train = train
        self.first_saccade_only = first_saccade_only
        self.keep_cmp = keep_compressed
        super().__init__(root, transform, target_transform, transforms)
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
        if not os.path.isfile(os.path.join(self._root, filename)):
            download_url(url=url, root=self._root, filename=filename)
        check_sha256(os.path.join(self._root, filename), sha256)

    def _uncompress(
        self, dp: IterDataPipe[Tuple[Any, BinaryIO]]
    ) -> IterDataPipe[Tuple[str, BinaryIO]]:
        # Stripping the archive from self._root.
        root = self._root
        folder = self._TRAIN_FOLDER if self.train else self._TEST_FOLDER
        # Joining root with a folder to contain the data.
        root = os.path.join(root, f"data_uncompressed/{folder}")
        if not os.path.isdir(root):
            os.makedirs(root)
            # Decompressing in root.
            def read_bin(fdata):
                return fdata.read()

            def filepath_fn(fpath):
                return os.path.join(root, fpath[fpath.find(folder) + len(folder) + 1 :])

            dp = Mapper(dp, read_bin, input_col=1)
            dp = Saver(dp, mode="wb", filepath_fn=filepath_fn)
            # Saving data to file.
            for x in dp:
                pass
        dp = FileLister(root, recursive=True)
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
