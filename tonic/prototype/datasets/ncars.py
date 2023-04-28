import os
from pathlib import Path
from typing import Callable, Iterator, Optional

from expelliarmus import Wizard
from expelliarmus.wizard.clib import event_t
from torchdata.datapipes.iter import FileLister, IterDataPipe

from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256


class NCARSFileReader(IterDataPipe[Sample]):
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
        folder_name = fname.split(os.sep)[-2]
        assert (
            folder_name == "background" or folder_name == "cars"
        ), f'Error, the folder name "{folder_name}" is wrong and cannot be associated to a label.'
        return 0 if folder_name == "background" else 1


class NCARS(Dataset):
    """`N-CARS <https://www.prophesee.ai/2018/03/13/dataset-n-cars/>`_

    This datasets needs 'expelliarmus' installed on the system. Events have "txyp" ordering.
    ::

        @article{Sironi_2018_CVPR,
          author = {Sironi, Amos and Brambilla, Manuele and Bourdis, Nicolas and Lagorce, Xavier and Benosman, Ryad},
          title = {HATS: Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification},
          booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          month = {June},
          year = {2018}
        }

    Parameters:
        root (string): Location to decompressed archive.
        train (bool): If True, uses training subset, otherwise testing subset.
    """

    _DTYPE = event_t
    _TRAIN_PATH = "n-cars_train"
    _TEST_PATH = "n-cars_test"
    sensor_size = dict(x=120, y=100, p=2)

    def __init__(
        self,
        root: os.PathLike,
        train: Optional[bool] = True,
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        self.train = train
        super().__init__(
            Path(root, self.__class__.__name__),
            False,
            skip_sha256_check,
        )
        assert (
            self._check_exists()
        ), "Error: the dataset files could not be found. You should download the dataset and manually extract it and, then, provide the path to the extracted archive as root."

    def __len__(self) -> int:
        return 7482 + 7940 if self.train else 4396 + 4211

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".dat")

    def _check_exists(self) -> bool:
        # Checking that train and test folders exist.
        ret = Path(self._root, self._TRAIN_PATH).is_dir()
        ret = ret and Path(self._root, self._TEST_PATH).is_dir()
        # Checking that some binary files are present.
        cnt = 0
        if ret:
            dp = FileLister(self._root, recursive=True).filter(self._filter)
            for f in dp:
                cnt += 1
        return ret and cnt > 0

    def _datapipe(self) -> IterDataPipe[Sample]:
        fpath = self._TRAIN_PATH if self.train else self._TEST_PATH
        fpath = os.path.join(self._root, fpath)
        dp = FileLister(str(fpath), recursive=True).filter(self._filter)
        # Reading data to structured NumPy array and integer target.
        dp = NCARSFileReader(dp)
        return dp
