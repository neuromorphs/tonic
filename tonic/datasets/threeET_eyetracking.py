import os
import warnings
from collections.abc import Callable
from typing import Any, Literal

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

    ``public_v1`` is the official public 16-training/2-validation benchmark
    published in the companion GitHub repository and is the default split
    definition. Its training and validation sets have no recording IDs in
    common, although different recordings from the same subject may occur in
    both sets. This public benchmark is distinct from the 22-recording
    experimental subset described in the paper.

    ``legacy_tonic_2023`` reproduces the split text files shipped in Tonic's
    original raw archive. That legacy split contains ``subj_18_vid_7`` in both
    training and validation and is provided only for historical reproducibility.
    Both versions use the same raw archive. Selecting ``legacy_tonic_2023``
    emits a warning because it contains complete-recording leakage.

    Parameters:
        save_to (string): Location to save files to on disk.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
        split_version (string, optional): Split definition to use. Defaults to
                                         ``public_v1``, which is
                                         recording-disjoint;
                                         ``legacy_tonic_2023`` reproduces the
                                         original Tonic archive manifests and
                                         emits a leakage warning.

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

    split_versions = ("public_v1", "legacy_tonic_2023")
    splits = ("train", "val")
    public_v1_recordings = {
        "train": (
            "subj_18_vid_2",
            "subj_10_vid_8",
            "subj_12_vid_1",
            "subj_12_vid_2",
            "subj_12_vid_9",
            "subj_13_vid_1",
            "subj_13_vid_2",
            "subj_14_vid_10",
            "subj_15_vid_2",
            "subj_21_vid_11",
            "subj_1_vid_1",
            "subj_1_vid_9",
            "subj_2_vid_4",
            "subj_6_vid_5",
            "subj_7_vid_18",
            "subj_8_vid_2",
        ),
        "val": ("subj_16_vid_1", "subj_18_vid_7"),
    }

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
        split_version: Literal["public_v1", "legacy_tonic_2023"] = "public_v1",
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        if split not in self.splits:
            raise ValueError(f"Invalid split {split!r}. Expected one of {self.splits}.")
        if split_version not in self.split_versions:
            raise ValueError(
                f"Invalid split_version {split_version!r}. "
                f"Expected one of {self.split_versions}."
            )

        if split_version == "legacy_tonic_2023":
            warnings.warn(
                "split_version='legacy_tonic_2023' reproduces archive manifests "
                "that contain 'subj_18_vid_7' in both training and validation. "
                "Use only for historical reproduction; public_v1 is the "
                "recording-disjoint official public benchmark.",
                UserWarning,
                stacklevel=2,
            )

        # if not exist, download from url
        if not self._check_exists():
            self.download()

        data_dir = self.location_on_system
        if split_version == "public_v1":
            filenames = list(self.public_v1_recordings[split])
        else:
            filenames = self.load_filenames(
                os.path.join(data_dir, f"{split}_files.txt")
            )

        self.split = split
        self.split_version = split_version
        self.filenames = filenames

        # Get the data file paths and target file paths
        self.data = [os.path.join(data_dir, "data", f + ".h5") for f in filenames]
        self.targets = [os.path.join(data_dir, "labels", f + ".txt") for f in filenames]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # get events from .h5 file
        with h5py.File(self.data[index], "r") as f:
            events = f["events"][:]
        # load the sparse labels
        with open(self.targets[index]) as f:
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

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    @staticmethod
    def load_filenames(path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
