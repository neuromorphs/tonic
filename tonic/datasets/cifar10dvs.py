import os
from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic.io import read_aedat4


class CIFAR10DVS(Dataset):
    """`CIFAR10-DVS <https://www.frontiersin.org/articles/10.3389/fnins.2017.00309/full>`_
    ::

        @article{li2017cifar10,
        title={Cifar10-dvs: an event-stream dataset for object classification},
        author={Li, Hongmin and Liu, Hanchao and Ji, Xiangyang and Li, Guoqi and Shi, Luping},
        journal={Frontiers in neuroscience},
        volume={11},
        pages={309},
        year={2017},
        publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    url = "https://figshare.com/ndownloader/files/38023437"

    filename = "CIFAR10DVS.zip"
    file_md5 = "ce3a4a0682dc0943703bd8f749a7701c"
    data_filename = [
        "airplane.zip",
        "automobile.zip",
        "bird.zip",
        "cat.zip",
        "deer.zip",
        "dog.zip",
        "frog.zip",
        "horse.zip",
        "ship.zip",
        "truck.zip",
    ]

    folder_name = ""
    dtype = np.dtype(
        [("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", bool)]
    )
    ordering = dtype.names
    sensor_size = (128, 128, 2)

    def __init__(
        self,
        save_to: str,
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

        # classes for CIFAR10DVS dataset

        classes = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }

        if not self._check_exists():
            self.download()
            for filename in self.data_filename:
                extract_archive(os.path.join(self.location_on_system, filename))

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("aedat4"):
                    self.data.append(os.path.join(path, file))
                    label_number = classes[os.path.basename(path)]
                    self.targets.append(label_number)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_aedat4(self.data[index])
        events.dtype.names = ["t", "x", "y", "p"]

        target = self.targets[index]

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
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(1000, ".aedat4")
        )
