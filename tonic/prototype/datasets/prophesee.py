import os
import pathlib
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from expelliarmus import Wizard
from expelliarmus.wizard.clib import event_t
from torchdata.datapipes.iter import FileLister, IterDataPipe

from .utils._dataset import Dataset, Sample


class MiniDatasetFileReader(IterDataPipe[Sample]):
    def __init__(
        self,
        dp: IterDataPipe[str],
    ) -> None:
        self.dp = dp
        self._wizard = Wizard(encoding="dat")

    def __iter__(self) -> Iterator[Sample]:
        for fname in self.dp:
            yield (
                self._wizard.read(fname),
                self._get_target(fname),
            )

    def _get_target(self, fname: str) -> int:
        return 0
        folder_name = fname.split(os.sep)[-2]
        assert (
            folder_name == "background" or folder_name == "cars"
        ), f'Error, the folder name "{folder_name}" is wrong and cannot be associated to a label.'
        return 0 if folder_name == "background" else 1


class MiniDataset(Dataset):
    """"""

    _DTYPE = event_t
    _URL = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
    _FILENAME = "mini_dataset.zip"
    _FOLDERNAME = "mini_dataset"
    _SHA256 = "a13fb1240c19f2e1dbf453cecbb9e0c3ac9a7a5ea3cfc5a4f88760fff4977449"
    _TRAIN_PATH = "train"
    _VALID_PATH = "val"
    _TEST_PATH = "test"

    sensor_size = (1280, 720, 2)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
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
        assert (
            self._check_exists()
        ), "Error: the dataset files could not be found. You should download the dataset and manually extract it and, then, provide the path to the extracted archive as root."

    def __len__(self) -> int:
        return {
            "train": 4,
            "valid": 1,
            "test": 1,
        }[self.split]

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".dat")

    def _check_exists(self) -> bool:
        # Checking that train and test folders exist.
        train_folder_exists = Path(
            self._root, self._FOLDERNAME, self._TRAIN_PATH
        ).is_dir()
        valid_folder_exists = Path(
            self._root, self._FOLDERNAME, self._VALID_PATH
        ).is_dir()
        test_folder_exists = Path(
            self._root, self._FOLDERNAME, self._TEST_PATH
        ).is_dir()

        # Checking that some binary files are present.
        if train_folder_exists and valid_folder_exists and test_folder_exists:
            dp = FileLister(self._root._str, recursive=True).filter(self._filter)
            if len(list(dp)) > 0:
                return True
        return False

    def _datapipe(self) -> IterDataPipe[Sample]:
        fpath = {
            "train": self._TRAIN_PATH,
            "valid": self._VALID_PATH,
            "test": self._TEST_PATH,
        }[self.split]
        fpath = os.path.join(self._root, self._FOLDERNAME, fpath)
        dp = FileLister(str(fpath), recursive=True).filter(self._filter)
        # Reading data to structured NumPy array and integer target.
        dp = MiniDatasetFileReader(dp)
        return dp
