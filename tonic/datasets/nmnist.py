import os
import numpy as np
from pathlib import Path

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class NMNIST(Dataset):
    """N-MNIST dataset <https://www.garrickorchard.com/datasets/n-mnist>. Events have (xytp) ordering.
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
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        first_saccade_only (bool): If True, only work with events of the first of three saccades. Results in about a third of the events overall.
    """

    base_url = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
    train_url = base_url + "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded"
    train_filename = "train.zip"
    train_md5 = "20959b8e626244a1b502305a9e6e2031"
    train_folder = "Train"
    test_url = base_url + "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded"
    test_filename = "test.zip"
    test_md5 = "69ca8762b2fe404d9b9bad1103e97832"
    test_folder = "Test"

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    sensor_size = (34, 34, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to,
        train=True,
        transform=None,
        target_transform=None,
        first_saccade_only=False,
    ):
        super(NMNIST, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.first_saccade_only = first_saccade_only

        if train:
            self.filename = self.train_filename
            self.url = self.train_url
            self.file_md5 = self.train_md5
            self.folder_name = self.train_folder
        else:
            self.filename = self.test_filename
            self.url = self.test_url
            self.file_md5 = self.test_md5
            self.folder_name = self.test_folder

        if not self._check_exists():
            self.download()

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    label_number = int(path[-1])
                    self.targets.append(label_number)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.data[index], dtype=self.dtype)
        if self.first_saccade_only:
            events = events[events["t"] < 1e5]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            10000, ".bin"
        )
