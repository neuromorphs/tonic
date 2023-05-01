import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    HttpReader,
    IterDataPipe,
    Mapper,
    Saver,
    ZipArchiveLoader,
)

from tonic.io import make_structured_array, read_mnist_file
from tonic.prototype.datasets.utils import HttpResource, OnlineResource

from .utils._dataset import Dataset2, Sample
from .utils._utils import check_sha256


class HSD(Dataset2):
    _DTYPE = np.dtype([("t", int), ("x", int), ("p", int)])
    _URL_ROOT = "https://zenkelab.org/datasets/"
    _URLS = {
        "train": f"{_URL_ROOT}shd_train.h5.zip",
        "test": f"{_URL_ROOT}shd_test.h5.zip",
    }
    _CHECKSUM = {
        "train": "d21bf17b71bededa52cc9134717b2bfee25dc4a99a94c1807be9a3c4f9697dc4",
        "test": "f50b1d1f614ed84ff92f34aa771c33b237944722455b52ce1ccd2eac7903812c",
    }

    sensor_size = dict(x=700, y=1, p=1)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = split
        super().__init__(
            root, dependencies=("scipy",)
        )  # skip_integrity_check=skip_integrity_check

    def _resources(self) -> List[OnlineResource]:
        return [
            HttpResource(self._URLS[self._split], sha256=self._CHECKSUM[self._split])
        ]

    def _datapipe(
        self, resource_dps: List[IterDataPipe]
    ) -> IterDataPipe[Dict[str, Any]]:
        return resource_dps
        images_dp, targets_dp = resource_dps
        if self._split == "train":
            targets_dp = Filter(
                targets_dp, path_comparator("name", "cars_train_annos.mat")
            )
        targets_dp = StanfordCarsLabelReader(targets_dp)
        dp = Zipper(images_dp, targets_dp)

        file = h5py.File(Path(self._root, self.data_filename), "r")

        for index in range(len(file["spikes/times"])):
            # adding artificial polarity of 1 and convert to microseconds
            events = make_structured_array(
                file["spikes/times"][index] * 1e6,
                file["spikes/units"][index],
                1,
                dtype=self.dtype,
            )
            target = file["labels"][index].astype(int)
            yield events, target

        return Mapper(dp, self._prepare_sample)

    def __len__(self) -> int:
        return 8_144 if self._split == "train" else 8_041

    def _check_exists(self):
        return super()._check_exists()


# class HSDFileReader(IterDataPipe[Sample]):
#     def __init__(
#         self,
#         dp: Union[IterDataPipe[str], IterDataPipe[Tuple[str, BinaryIO]]],
#         dtype: Optional[np.dtype] = np.dtype(
#             [("x", int), ("y", int), ("t", int), ("p", int)]
#         ),
#         keep_compressed: Optional[bool] = False,
#     ) -> None:
#         self.dp = dp
#         self.dtype = dtype
#         self.keep_cmp = keep_compressed

#     def __iter__(self) -> Iterator[Sample]:
#         for fname, fdata in self.dp:
#                 yield (
#                     read_mnist_file(fdata, self.dtype, is_stream=self.keep_cmp),
#                     self._get_target(fname),
#                 )

#     def _get_target(self, fname: str) -> int:
#         return int(fname.split(os.sep)[-2])


# ## A dataset class for HSD that inherits from Dataset and uses torchdata.datapipes.iter
# class HSD(Dataset):
#     """ """
#     _DTYPE=np.dtype([("t", int), ("x", int), ("p", int)])
#     _URL = "https://zenkelab.org/datasets/"
#     _TEST_ZIP = "shd_test.h5.zip"
#     _TRAIN_ZIP = "shd_train.h5.zip"
#     _TEST_MD5 = "1503a5064faa34311c398fb0a1ed0a6f"
#     _TRAIN_MD5 = "f3252aeb598ac776c1b526422d90eecb"

#     sensor_size = dict(x=700, y=1, p=1)
#     dtype = np.dtype([("t", int), ("x", int), ("p", int)])

#     def __init__(
#         self,
#         root: Union[str, Path],
#         split: str = "train",
#         skip_sha256_check: Optional[bool] = True,
#     ) -> None:
#         self.split = split
#         super().__init__(
#             root=Path(root, self.__class__.__name__),
#             skip_sha256_check=skip_sha256_check,
#             dependencies="h5py",
#         )
#         if not self._check_exists():
#             self.download()

#     def _check_exists(self) -> bool:
#         return (self._root / "HSD").exists()

#     def _datapipe(self) -> IterDataPipe[Sample]:
#         if self.split=="train":
#             self.url = self._URL + self._TRAIN_ZIP
#             self.filename = self._TRAIN_ZIP
#             self.file_md5 = self._TRAIN_MD5
#         else:
#             self.url = self._URL + self._TEST_ZIP
#             self.filename = self._TEST_ZIP
#             self.file_md5 = self._TEST_MD5
#         self.data_filename = self.filename[:-4]


#     def __len__(self) -> int:
#         return len(list(self._datapipe()))
