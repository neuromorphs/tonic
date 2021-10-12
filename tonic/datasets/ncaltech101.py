import os
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class NCALTECH101(Dataset):
    """N-CALTECH101 dataset <https://www.garrickorchard.com/datasets/n-caltech101>. Events have (xytp) ordering.
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
        download (bool): Choose to download data or verify existing files. If True and a file with the same
                    name and correct hash is already in the directory, download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "https://www.dropbox.com/sh/iuv7o3h2gv6g4vd/AADYPdhIBK7g_fPCLKmG6aVpa?dl=1"
    archive_filename = "N-Caltech101-archive.zip"
    archive_md5 = "989af2c704103341d616b748b5daa70c"
    file_md5 = "66201824eabb0239c7ab992480b50ba3"
    filename = "Caltech101.zip"
    folder_name = "Caltech101"

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, download=True, transform=None, target_transform=None):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.location_on_system = os.path.join(save_to, "ncaltech-101/")
        self.samples = []
        self.targets = []

        if download:
            self.download()

        else:
            if not check_integrity(
                os.path.join(self.location_on_system, self.filename), self.file_md5
            ):
                raise RuntimeError(
                    "Dataset not found or corrupted."
                    + " You can use download=True to download it."
                )

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("bin"):
                    self.samples.append(path + "/" + file)
                    label_number = os.path.basename(path)
                    self.targets.append(label_number)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.samples[index], dtype=self.dtype)
        target = self.targets[index]
        events["x"] -= events["x"].min()
        events["y"] -= events["y"].min()
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        download_and_extract_archive(
            self.url,
            self.location_on_system,
            filename=self.archive_filename,
            md5=self.archive_md5,
        )
        extract_archive(os.path.join(self.location_on_system, self.filename))
