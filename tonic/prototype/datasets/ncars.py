from .utils._dataset import Dataset, Sample
from .utils._utils import check_sha256
import os
from expelliarmus import Wizard
from typing import Optional, Union, Tuple, Iterator, Any, Callable
import numpy as np
import pathlib
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
)


class NCARSFileReader(IterDataPipe[Sample]):
    def __init__(
        self,
        dp: IterDataPipe[str],
        dtype: Optional[np.dtype] = np.dtype(
            [("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.uint8)]
        )
    ) -> None:
        self.dp = dp
        self.dtype = dtype
        self._wizard = Wizard(encoding="dat", dtype=dtype)

    def __iter__(self) -> Iterator[Sample]:
        for fname in self.dp:
            yield (
                self._wizard.read(fname),
                self._get_target(fname),
            )

    def _get_target(self, fname: str) -> int:
        folder_name = fname.split(os.sep)[-2]
        assert folder_name=="background" or folder_name=="cars", f"Error, the folder name \"\{folder_name}\" is wrong and cannot be associated to a label." 
        return 0 if folder_name=="background" else 1


class NCARS(Dataset):
    """N-CARS <https://www.prophesee.ai/2018/03/13/dataset-n-cars/>

    Events have (txyp) ordering.
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
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and labels at the same time.
    """

    _DTYPE = np.dtype([("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.uint8)])
    _TRAIN_PATH = "n-cars_train"
    _TEST_PATH = "n-cars_test"
    sensor_size = (120, 100, 2)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        train: Optional[bool] = True,
        skip_sha256_check: Optional[bool] = True,
    ) -> None:
        self.train = train
        super().__init__(
            root,
            transform,
            target_transform,
            transforms,
            False,
            skip_sha256_check,
        )

    def __len__(self) -> int:
        return 7482+7940 if self.train else 4396+4211

    def _filter(self, fname: str) -> bool:
        return fname.endswith(".dat")

    def _check_exists(self) -> bool:
        # Checking that train and test folders exist.
        ret = pathlib.Path(self._root, _TRAIN_PATH).is_dir()
        ret = ret and pathlib.Path(self._root, _TEST_PATH).is_dir()
        # Checking that some binary files are present.
        cnt = 0
        if ret:
            dp = FileLister(self._root, recursive=True).filter(self._filter)
            for f in dp: 
                cnt += 1
        return ret and cnt>0

    def _datapipe(self) -> IterDataPipe[Sample]:
        fpath = self._TRAIN_PATH if self.train else self._TEST_PATH
        fpath = os.path.join(self._root, fpath)
        dp = FileLister(str(fpath), recursive=True).filter(self._filter)
        # Reading data to structured NumPy array and integer target.
        dp = NCARSFileReader(dp, dtype=self._DTYPE)
        return dp
