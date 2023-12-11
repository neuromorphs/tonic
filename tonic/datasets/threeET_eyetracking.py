import os
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np

from tonic.dataset import Dataset
from tonic.io import make_structured_array


class ThreeET_Eyetracking(Dataset):
    """3ET DVS eye tracking `3ET <https://github.com/qinche106/cb-convlstm-eyetracking>`_
    ::

        @article{chen20233et,
            title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
            author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
            journal={arXiv preprint arXiv:2308.11771},
            year={2023}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.

    Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    url = "https://dl.dropboxusercontent.com/s/1hyer8egd8843t9/ThreeET_Eyetracking.zip?dl=0"
    filename = "ThreeET_Eyetracking.zip"
    file_md5 = "b6c652b06fdfd85721f39e2dbe12f4e8"

    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        # if not exist, download from url
        if not self._check_exists():
            self.download()

        data_dir = os.path.join(save_to, "ThreeET_Eyetracking")
        # Load filenames from the provided lists
        if split == "train":
            filenames = self.load_filenames(os.path.join(data_dir, "train_files.txt"))
        elif split == "val":
            filenames = self.load_filenames(os.path.join(data_dir, "val_files.txt"))
        else:
            raise ValueError("Invalid split name")

        # Get the data file paths and target file paths
        self.data = [os.path.join(data_dir, "data", f + ".h5") for f in filenames]
        self.targets = [os.path.join(data_dir, "labels", f + ".txt") for f in filenames]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # get events from .h5 file
        with h5py.File(self.data[index], "r") as f:
            events = f["events"][:]
        # load the sparse labels
        with open(self.targets[index], "r") as f:
            target = np.array(
                [line.strip().split() for line in f.readlines()], np.float64
            )

        events = make_structured_array(
            events[:, 0],  # time in us
            events[:, 1],  # x
            events[:, 2],  # y
            events[:, 3],  # polarity in 1 or 0
            dtype=self.dtype,
        )

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    def load_filenames(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
