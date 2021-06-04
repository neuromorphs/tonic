import os
import numpy as np
import h5py
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class NTIDIGITS(VisionDataset):
    """N-TIDIGITS <https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit> data set.

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    #     url = "https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"
    url = "https://www.neuromorphic-vision.com/public/downloads/n-tidigits.hdf5"
    file_md5 = "360a2d11e5429555c9197381cf6b58e0"
    filename = "n-tidigits.hdf5"

    sensor_size = (64,)
    ordering = "tx"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None,
    ):
        super(NTIDIGITS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.location_on_system = save_to

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

    def __getitem__(self, index):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        if self.train:
            target = bytes.decode(file["train_labels"][index])
            timestamps = np.array(file["train_timestamps/" + target])
            addresses = np.array(file["train_addresses/" + target])
        else:
            target = bytes.decode(file["test_labels"][index])
            timestamps = np.array(file["test_timestamps/" + target])
            addresses = np.array(file["test_addresses/" + target])
        events = np.vstack((timestamps, addresses)).T
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        if self.train:
            return len(file["train_labels"])
        else:
            return len(file["test_labels"])

    def download(self):
        download_url(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
