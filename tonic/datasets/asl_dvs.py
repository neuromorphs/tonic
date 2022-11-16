import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.io as scio

from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic.io import make_structured_array


class ASLDVS(Dataset):
    """`ASL-DVS <https://github.com/PIX2NVS/NVS2Graph>`_
    ::

        @inproceedings{bi2019graph,
            title={Graph-based Object Classification for Neuromorphic Vision Sensing},
            author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
            booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
            year={2019},
            organization={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    url = "https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=1"
    filename = "ASLDVS.zip"
    file_md5 = "33f8b87bf45edc0bfed0de41822279b9"
    folder_name = ""

    classes = [chr(letter) for letter in range(97, 123)]  # generate alphabet
    int_classes = dict(zip(classes, range(len(classes))))
    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

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

        if not self._check_exists():
            self.download()
            # extract zips within zip
            for path, dirs, files in os.walk(self.location_on_system):
                dirs.sort()
                for file in files:
                    if file.startswith("Yin") and file.endswith("zip"):
                        extract_archive(os.path.join(self.location_on_system, file))

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    self.data.append(path + "/" + file)
                    self.targets.append(self.int_classes[path[-1]])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        events, target = scio.loadmat(self.data[index]), self.targets[index]
        events = make_structured_array(
            events["ts"],
            events["x"],
            self.sensor_size[1] - 1 - events["y"],
            events["pol"],
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
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(100800, ".mat")
        )
