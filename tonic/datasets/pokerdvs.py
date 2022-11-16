import os
from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset


class POKERDVS(Dataset):
    """`POKER-DVS <http://www2.imse-cnm.csic.es/caviar/POKERDVS.html>`_

    Events have (txyp) ordering.
    ::

        @article{serrano2015poker,
          title={Poker-DVS and MNIST-DVS. Their history, how they were made, and other details},
          author={Serrano-Gotarredona, Teresa and Linares-Barranco, Bernab{\'e}},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={481},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    train_filename = "pips_train.tar.gz"
    test_filename = "pips_test.tar.gz"
    train_url = base_url + train_filename
    test_url = base_url + test_filename
    train_md5 = "412bcfb96826e4fcb290558e8c150aae"
    test_md5 = "eef2bf7d0d3defae89a6fa98b07c17af"

    classes = ["cl", "he", "di", "sp"]
    int_classes = dict(zip(classes, range(4)))
    sensor_size = (35, 35, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
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

        self.train = train

        if train:
            self.url = self.train_url
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "pips_train"
        else:
            self.url = self.test_url
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "pips_test"

        if not self._check_exists():
            self.download()

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.targets.append(self.int_classes[path[-2:]])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = self.data[index], self.targets[index]
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
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
            and self._folder_contains_at_least_n_files_of_type(20, ".npy")
        )
