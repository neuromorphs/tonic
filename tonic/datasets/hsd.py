import os
import numpy as np
import h5py
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class HSD(VisionDataset):
    """Heidelberg Spiking Datasets <https://arxiv.org/abs/1910.07407> data set contains the Spiking Heidelberg Dataset (SHD) and the Spiking Speech Commands dataset (SSC)"""

    base_url = "https://zenkelab.org/datasets/"
    sensor_size = (700,)
    ordering = "tx"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None
    ):
        super(HSD, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.location_on_system = save_to

        if train:
            self.url = self.base_url + self.train_zip
            self.zipfile = self.train_zip
            self.file_md5 = self.train_md5
        else:
            self.url = self.base_url + self.test_zip
            self.zipfile = self.test_zip
            self.file_md5 = self.test_md5
        self.filename = self.zipfile[:-4]

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.zipfile), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        self.classes = file["extra/keys"]

    def __getitem__(self, index):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        events = np.vstack((file["spikes/times"][index], file["spikes/units"][index])).T
        target = file["labels"][index].astype(np.int)
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        return len(file["labels"])

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.zipfile, md5=self.file_md5
        )


class SHD(HSD):
    """Spiking Heidelberg Dataset. One of two Heidelberg Spiking Datasets <https://arxiv.org/abs/1910.07407>.

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    test_zip = "shd_test.h5.zip"
    train_zip = "shd_train.h5.zip"
    test_md5 = "1503a5064faa34311c398fb0a1ed0a6f"
    train_md5 = "f3252aeb598ac776c1b526422d90eecb"


class SSC(HSD):
    """Spiking Speech Commands dataset. One of two Heidelberg Spiking Datasets <https://arxiv.org/abs/1910.07407>.

    Args:
        save_to (string): Location to save files to on disk.
        split (string): One of 'train', 'test' or 'valid'
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
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

    def __init__(
        self,
        save_to,
        split="train",
        download=True,
        transform=None,
        target_transform=None,
    ):
        super(HSD, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.location_on_system = save_to

        if split == "train":
            self.url = self.base_url + self.train_zip
            self.zipfile = self.train_zip
            self.file_md5 = self.train_md5
        elif split == "test":
            self.url = self.base_url + self.test_zip
            self.zipfile = self.test_zip
            self.file_md5 = self.test_md5
        elif split == "valid":
            self.url = self.base_url + self.valid_zip
            self.zipfile = self.valid_zip
            self.file_md5 = self.valid_md5
        self.filename = self.zipfile[:-4]

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.zipfile), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        self.classes = file["extra/keys"]
