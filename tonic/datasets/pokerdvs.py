import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class POKERDVS(VisionDataset):
    """POKER DVS <http://www2.imse-cnm.csic.es/caviar/POKERDVS.html> data set

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
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
    sensor_size = (35, 35)
    ordering = "txyp"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None
    ):
        super(POKERDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.train = train
        self.location_on_system = save_to
        self.data = []
        self.targets = []

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

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file_path = self.location_on_system + "/" + self.folder_name
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(path + "/" + file))
                    self.targets.append(self.int_classes[path[-2:]])

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]
        events = events.astype(np.float)
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
