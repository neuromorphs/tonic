import os
import numpy as np
import h5py
from tonic.dataset import Dataset
from tonic.io import make_structured_array


class HSD(Dataset):
    """Heidelberg Spiking Dataset <https://arxiv.org/abs/1910.07407> contains the Spiking Heidelberg Dataset (SHD)
    and the Spiking Speech Commands dataset (SSC)."""

    base_url = "https://zenkelab.org/datasets/"
    sensor_size = (700, 1, 1)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    def __getitem__(self, index):
        file = h5py.File(os.path.join(self.location_on_system, self.data_filename), "r")
        # adding artificial polarity of 1 and convert to microseconds
        events = make_structured_array(
            file["spikes/times"][index] * 1e6,
            file["spikes/units"][index],
            1,
            dtype=self.dtype,
        )
        target = file["labels"][index].astype(int)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        file = h5py.File(os.path.join(self.location_on_system, self.data_filename), "r")
        return len(file["labels"])

    def _check_exists(self):
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(1, ".h5")
        )


class SHD(HSD):
    """`Spiking Heidelberg Digits <https://arxiv.org/abs/1910.07407>`_
    ::

        @article{cramer2020heidelberg,
          title={The heidelberg spiking data sets for the systematic evaluation of spiking neural networks},
          author={Cramer, Benjamin and Stradmann, Yannik and Schemmel, Johannes and Zenke, Friedemann},
          journal={IEEE Transactions on Neural Networks and Learning Systems},
          year={2020},
          publisher={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will put files in an 'hsd' subfolder.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    test_zip = "shd_test.h5.zip"
    train_zip = "shd_train.h5.zip"
    test_md5 = "1503a5064faa34311c398fb0a1ed0a6f"
    train_md5 = "f3252aeb598ac776c1b526422d90eecb"
    folder_name = ""

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(HSD, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if train:
            self.url = self.base_url + self.train_zip
            self.filename = self.train_zip
            self.file_md5 = self.train_md5
        else:
            self.url = self.base_url + self.test_zip
            self.filename = self.test_zip
            self.file_md5 = self.test_md5
        self.data_filename = self.filename[:-4]

        if not self._check_exists():
            self.download()

        file = h5py.File(os.path.join(self.location_on_system, self.data_filename), "r")
        self.classes = file["extra/keys"][()]


class SSC(HSD):
    """`Spiking Speech Commands <https://arxiv.org/abs/1910.07407>`_
    ::

        @article{cramer2020heidelberg,
          title={The heidelberg spiking data sets for the systematic evaluation of spiking neural networks},
          author={Cramer, Benjamin and Stradmann, Yannik and Schemmel, Johannes and Zenke, Friedemann},
          journal={IEEE Transactions on Neural Networks and Learning Systems},
          year={2020},
          publisher={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will put files in an 'hsd' subfolder.
        split (string): One of 'train', 'test' or 'valid'.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    test_zip = "ssc_test.h5.zip"
    train_zip = "ssc_train.h5.zip"
    valid_zip = "ssc_valid.h5.zip"
    test_md5 = "a35ff1e9cffdd02a20eb850c17c37748"
    train_md5 = "d102be95e7144fcc0553d1f45ba94170"
    valid_md5 = "b4eee3516a4a90dd0c71a6ac23a8ae43"
    folder_name = ""

    def __init__(self, save_to, split="train", transform=None, target_transform=None):
        save_to = os.path.join(save_to, self.__class__.__name__)
        super(HSD, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if split == "train":
            self.url = self.base_url + self.train_zip
            self.filename = self.train_zip
            self.file_md5 = self.train_md5
        elif split == "test":
            self.url = self.base_url + self.test_zip
            self.filename = self.test_zip
            self.file_md5 = self.test_md5
        elif split == "valid":
            self.url = self.base_url + self.valid_zip
            self.filename = self.valid_zip
            self.file_md5 = self.valid_md5
        self.data_filename = self.filename[:-4]

        if not self._check_exists():
            self.download()

        file = h5py.File(os.path.join(self.location_on_system, self.data_filename), "r")
        self.classes = file["extra/keys"][()]
