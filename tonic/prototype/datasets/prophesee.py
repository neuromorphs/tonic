import os
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from expelliarmus import Wizard
from expelliarmus.wizard.clib import event_t
from torchdata.datapipes.iter import FileLister, IterDataPipe

from tonic.download_utils import download_url

from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256


class Gen4AutomotiveDetectionMiniFileReader(IterDataPipe[Sample]):
    def __init__(
        self,
        dp: IterDataPipe[str],
    ) -> None:
        self.dp = dp
        self._wizard = Wizard(encoding="dat")

    def __iter__(self) -> Iterator[Sample]:
        for data_file_path, label_file_path in self.dp:
            yield (
                self._wizard.read(data_file_path),
                np.load(label_file_path),
            )


class Gen4AutomotiveDetectionMini(Dataset):
    """`Gen4 Automotive Detection <https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-
    based-dataset/>`_

    This datasets needs 'expelliarmus' installed on the system. Events have "txyp" ordering.
    ::

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
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and labels at the same time.
    """

    _DTYPE = event_t
    _URL = "https://dataset.prophesee.ai/index.php/s/ScqMu02G5pdYKPh/download"
    _FILENAME = "mini_dataset.zip"
    _FOLDERNAME = "mini_dataset"
    _SHA256 = "a13fb1240c19f2e1dbf453cecbb9e0c3ac9a7a5ea3cfc5a4f88760fff4977449"
    _TRAIN_FOLDER = "train"
    _VALID_FOLDER = "val"
    _TEST_FOLDER = "test"

    sensor_size = (1280, 720, 2)

    def __init__(
        self,
        root: os.PathLike,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        split: str = "train",
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        self.split = split
        super().__init__(
            Path(root, self.__class__.__name__),
            transform,
            target_transform,
            transforms,
            False,
            skip_sha256_check,
        )
        if not self._check_exists():
            self._download()

    def __len__(self) -> int:
        return {
            "train": 4,
            "valid": 1,
            "test": 1,
        }[self.split]

    def _dat_filter(self, fname: str) -> bool:
        return fname.endswith(".dat")

    def _label_filter(self, fname: str) -> bool:
        return fname.endswith(".npy")

    def _check_exists(self) -> bool:
        base_path = Path(self._root, self._FOLDERNAME)
        train_folder_exists = (base_path / self._TRAIN_FOLDER).is_dir()
        valid_folder_exists = (base_path / self._VALID_FOLDER).is_dir()
        test_folder_exists = (base_path / self._TEST_FOLDER).is_dir()

        # Checking that some binary files are present.
        if train_folder_exists and valid_folder_exists and test_folder_exists:
            dp = FileLister(str(self._root), recursive=True).filter(self._dat_filter)
            if len(list(dp)) > 0:
                return True
        return False

    def _download(self):
        download_url(url=self._URL, root=self._root, filename=self._FILENAME)
        check_sha256(Path(self._root, self._FILENAME), self._SHA256)

    def _datapipe(self) -> IterDataPipe[Sample]:
        split_folder = {
            "train": self._TRAIN_FOLDER,
            "valid": self._VALID_FOLDER,
            "test": self._TEST_FOLDER,
        }[self.split]
        fpath = Path(self._root, self._FOLDERNAME, split_folder)
        data_dp = FileLister(str(fpath), recursive=True).filter(self._dat_filter)
        label_dp = FileLister(str(fpath), recursive=True).filter(self._label_filter)
        return Gen4AutomotiveDetectionMiniFileReader(zip(data_dp, label_dp))
