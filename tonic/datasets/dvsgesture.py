import os
from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset


class DVSGesture(Dataset):
    """`IBM DVS Gestures <http://research.ibm.com/dvsgesture/>`_

    .. note::  This is (exceptionally) a preprocessed version of the original dataset,
               where recordings that originally contained multiple labels have already
               been cut into respective samples. Also temporal precision is reduced to ms.

    ::

        @inproceedings{amir2017low,
          title={A low power, fully event-based gesture recognition system},
          author={Amir, Arnon and Taba, Brian and Berg, David and Melano, Timothy and McKinstry, Jeffrey and Di Nolfo, Carmelo and Nayak, Tapan and Andreopoulos, Alexander and Garreau, Guillaume and Mendoza, Marcela and others},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          pages={7243--7252},
          year={2017}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    test_url = "https://figshare.com/ndownloader/files/38020584"
    train_url = "https://figshare.com/ndownloader/files/38022171"
    test_md5 = "56070e45dadaa85fff82e0fbfbc06de5"
    train_md5 = "3a8f0d4120a166bac7591f77409cb105"
    test_filename = "ibmGestureTest.tar.gz"
    train_filename = "ibmGestureTrain.tar.gz"
    classes = [
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    ]

    sensor_size = (128, 128, 2)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
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
            self.folder_name = "ibmGestureTrain"
        else:
            self.url = self.test_url
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "ibmGestureTest"

        if not self._check_exists():
            self.download()

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(path + "/" + file)
                    self.targets.append(int(file[:-4]))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = np.load(self.data[index])
        events[:, 3] *= 1000  # convert from ms to us
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
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
            and self._folder_contains_at_least_n_files_of_type(100, ".npy")
        )
