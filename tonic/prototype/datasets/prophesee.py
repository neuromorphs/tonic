import os
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
from expelliarmus import Wizard
from expelliarmus.wizard.clib import event_t
from torchdata.datapipes.iter import FileLister, IterDataPipe, Shuffler

from tonic.download_utils import download_url

from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256


class AutomotiveDetectionFileReader(IterDataPipe[Sample]):
    def __init__(self, dp: IterDataPipe[str]) -> None:
        self.dp = dp
        self._wizard = Wizard(encoding="dat")

    def __iter__(self) -> Iterator[Sample]:
        for data_file_path, label_file_path in self.dp:
            yield (
                self._wizard.read(data_file_path),
                np.load(label_file_path),
            )


class AutomotiveDetectionBaseClass(Dataset):
    """Base class for Automotive Detection datasets."""

    _DTYPE = event_t
    _TRAIN_FOLDER = "train"
    _VALID_FOLDER = "val"
    _TEST_FOLDER = "test"

    def __init__(
        self,
        root: os.PathLike,
        split: str = "train",
        skip_sha256_check: Optional[bool] = True,
        shuffle=False,
    ) -> None:
        self.split = split
        self.do_shuffle = shuffle
        super().__init__(
            root=root,
            keep_compressed=False,
            skip_sha256_check=skip_sha256_check,
        )

    def _dat_filter(self, fname: str) -> bool:
        return fname.endswith(".dat")

    def _label_filter(self, fname: str) -> bool:
        return fname.endswith(".npy")

    def _check_exists(self) -> bool:
        base_path = Path(self._root, self._FOLDERNAME)
        split_mapping = {
            "train": self._TRAIN_FOLDER,
            "valid": self._VALID_FOLDER,
            "test": self._TEST_FOLDER,
        }
        split_folder = base_path / split_mapping[self.split]
        folder_exists = split_folder.is_dir()

        # Checking that some binary files are present.
        file_dp = FileLister(str(split_folder), recursive=True).filter(self._dat_filter)
        return True if folder_exists and len(list(file_dp)) > 0 else False

    def _datapipe(self) -> IterDataPipe[Sample]:
        split_folder = {
            "train": self._TRAIN_FOLDER,
            "valid": self._VALID_FOLDER,
            "test": self._TEST_FOLDER,
        }[self.split]
        fpath = Path(self._root, self._FOLDERNAME, split_folder)
        data_dp = FileLister(str(fpath), recursive=True).filter(self._dat_filter)
        label_dp = FileLister(str(fpath), recursive=True).filter(self._label_filter)

        dp = zip(data_dp, label_dp)
        if self.do_shuffle:
            dp = Shuffler(dp, buffer_size=1_000_000)
        return AutomotiveDetectionFileReader(dp)


class Gen1AutomotiveDetection(AutomotiveDetectionBaseClass):
    """`Gen1 Automotive Detection Dataset <https://www.prophesee.ai/2020/01/24/prophesee-
    gen1-automotive-detection-dataset/>`_

    To download the data, you'll need to agree to Prophesee's Terms and Conditions.

    Then, the steps to acquire the data can be as follows:

    Download the torrent file for the dataset::

        wget https://dataset.prophesee.ai/index.php/s/uE0QGLaFAEQnPwy/download\?path\=%2F\&files\=ATIS%20Automotive%20Detection%20Dataset.torrent
        -O Gen1Prophesee.torrent

    Download the data using peer-to-peer connections. On Linux this can be done using `aria2c` on the command line::

        aria2c Gen1Prophesee.torrent

    This will download several 7z archives for training and testing. We'll need to unpack them manually by looping over the 7z files and feeding them to 7z::
    
        sudo apt-get install p7zip-full
        for i in *.7z; do 7z x $i; done

    Parameters:
        root (string): Location to decompressed archive.
        split (str): Can be 'train' (default), 'valid' or 'test'.
        shuffle (bool): If True, the dataset will be shuffled randomly.
    """

    _FOLDERNAME = "detection_dataset_duration_60s_ratio_1.0"
    sensor_size = dict(x=304, y=240, p=2)
    class_map = {
        0: "car",
        1: "pedestrian",
    }

    def __init__(self, root: os.PathLike, split: str = "train", shuffle: bool = False):
        super().__init__(root=root, split=split, shuffle=shuffle)
        if not self._check_exists():
            raise RuntimeError(
                "You need to download and extract the dataset manually. See the Tonic documentation for more details."
            )

    def __len__(self) -> int:
        return {
            "train": 1459,
            "valid": 429,
            "test": 470,
        }[self.split]


class Gen4AutomotiveDetectionMini(AutomotiveDetectionBaseClass):
    """`Gen4 Automotive Detection <https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-
    based-dataset/>`_

    This datasets needs 'expelliarmus' installed on the system. Events have "txyp" ordering.::

        @article{de2020large,
          title={A large scale event-based detection dataset for automotive},
          author={De Tournemire, Pierre and Nitti, Davide and Perot, Etienne and Migliore, Davide and Sironi, Amos},
          journal={arXiv preprint arXiv:2001.08499},
          year={2020}
        }

    .. note:: The hosting server is very flaky and often interrupts the download before it is completed. If you end up with anything smaller than 23GB on disk, delete and try again.

    Parameters:
        root (string): Location to decompressed archive.
        split (str): Can be 'train' (default), 'valid' or 'test'.
        shuffle (bool): If True, the dataset will be shuffled randomly.
    """

    _URL = "https://dataset.prophesee.ai/index.php/s/ScqMu02G5pdYKPh/download"
    _FILENAME = "mini_dataset.zip"
    _FOLDERNAME = "mini_dataset"
    _SHA256 = "a13fb1240c19f2e1dbf453cecbb9e0c3ac9a7a5ea3cfc5a4f88760fff4977449"

    sensor_size = dict(x=1280, y=720, p=2)
    class_map = {
        0: "pedestrian",
        1: "two wheeler",
        2: "car",
        3: "truck",
        4: "bus",
        5: "traffic sign",
        6: "traffic light",
    }

    def __init__(
        self,
        root: os.PathLike,
        split: str = "train",
        shuffle: bool = False,
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        super().__init__(
            root=Path(root, self.__class__.__name__),
            split=split,
            shuffle=shuffle,
            skip_sha256_check=skip_sha256_check,
        )
        if not self._check_exists():
            self._download()

    def __len__(self) -> int:
        return {
            "train": 4,
            "valid": 1,
            "test": 1,
        }[self.split]

    def _download(self):
        download_url(url=self._URL, root=self._root, filename=self._FILENAME)
        check_sha256(Path(self._root, self._FILENAME), self._SHA256)


class Gen4Automotive(AutomotiveDetectionBaseClass):
    """`Gen4 Automotive Detection <https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-
    based-dataset/>`_

    This datasets needs 'expelliarmus' installed on the system. Events have "txyp" ordering.::

        @article{de2020large,
          title={A large scale event-based detection dataset for automotive},
          author={De Tournemire, Pierre and Nitti, Davide and Perot, Etienne and Migliore, Davide and Sironi, Amos},
          journal={arXiv preprint arXiv:2001.08499},
          year={2020}
        }

    To download the data, you'll need to agree to Prophesee's Terms and Conditions.

    Then, the steps to acquire the data can be as follows:

    Download the torrent file for the dataset::

        wget https://dataset.prophesee.ai/index.php/s/8HY0Bv4mOU4RzBm/download?path=%2F&files=Large_Automotive_Detection_Dataset.torrent
        -O Gen4Prophesee.torrent

    Download the data using peer-to-peer connections. On Linux this can be done using `aria2c` on the command line::

        aria2c Gen4Prophesee.torrent

    This will download several 7z archives for training, validation and testing. We'll need to unpack them manually by looping over the 7z files and feeding them to 7z::

        sudo apt-get install p7zip-full
        for i in *.7z; do 7z x $i; done

    Parameters:
        root (string): Location to decompressed archive.
        split (str): Can be 'train' (default), 'valid' or 'test'.
        shuffle (bool): If True, the dataset will be shuffled randomly.
    """

    _FOLDERNAME = "Large_Automotive_Detection_Dataset"
    sensor_size = dict(x=1280, y=720, p=2)
    class_map = {
        0: "pedestrian",
        1: "two wheeler",
        2: "car",
        3: "truck",
        4: "bus",
        5: "traffic sign",
        6: "traffic light",
    }

    def __init__(
        self,
        root: os.PathLike,
        split: str = "train",
        shuffle: bool = False,
    ) -> None:
        super().__init__(root=root, split=split, shuffle=shuffle)
        if not self._check_exists():
            raise RuntimeError(
                "You need to download and extract the dataset manually. See the Tonic documentation for more details."
            )

    def __len__(self) -> int:
        return {
            "train": 4,
            "valid": 1,
            "test": 1,
        }[self.split]


class Gen4Downsampled(Dataset):
    _FOLDERNAME = "Gen 4 Histograms"
    _TRAIN_FOLDER = "train"
    _VALID_FOLDER = "val"
    _TEST_FOLDER = "test"

    def __init__(
        self,
        root: os.PathLike,
        split: str = "train",
        skip_sha256_check: Optional[bool] = True,
        shuffle=False,
    ) -> None:
        self.split = split
        self.do_shuffle = shuffle
        super().__init__(
            root=root,
            keep_compressed=False,
            skip_sha256_check=skip_sha256_check,
        )

    def _dat_filter(self, fname: str) -> bool:
        return fname.endswith(".h5")

    def _label_filter(self, fname: str) -> bool:
        return fname.endswith(".npy")

    def _check_exists(self) -> bool:
        base_path = Path(self._root, self._FOLDERNAME)
        split_mapping = {
            "train": self._TRAIN_FOLDER,
            "valid": self._VALID_FOLDER,
            "test": self._TEST_FOLDER,
        }
        split_folder = base_path / split_mapping[self.split]
        folder_exists = split_folder.is_dir()

        # Checking that some binary files are present.
        file_dp = FileLister(str(split_folder), recursive=True).filter(self._dat_filter)
        return True if folder_exists and len(list(file_dp)) > 0 else False

    def _datapipe(self) -> IterDataPipe[Sample]:
        split_folder = {
            "train": self._TRAIN_FOLDER,
            "valid": self._VALID_FOLDER,
            "test": self._TEST_FOLDER,
        }[self.split]
        fpath = Path(self._root, self._FOLDERNAME, split_folder)
        data_dp = FileLister(str(fpath), recursive=True).filter(self._dat_filter)
        label_dp = FileLister(str(fpath), recursive=True).filter(self._label_filter)

        dp = zip(data_dp, label_dp)
        if self.do_shuffle:
            dp = Shuffler(dp, buffer_size=1_000_000)
        for data_file_path, label_file_path in dp:
            yield (
                h5py.File(data_file_path)["data"][()],
                np.load(label_file_path),
            )

    def __len__(self) -> int:
        return {
            "train": 705,
            "valid": 131,
            "test": 119,
        }[self.split]
