import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class NMNIST(VisionDataset):
    """N-MNIST <https://www.garrickorchard.com/datasets/n-mnist> data set.

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        first_saccade_only (bool): If True, only work with events of the first of three saccades. Results in about a third of the events overall.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=1"
    archive_filename = "nmnist-archive.zip"
    archive_md5 = "c5b12b1213584bd3fe976b55fe43c835"
    train_md5 = "20959b8e626244a1b502305a9e6e2031"
    test_md5 = "69ca8762b2fe404d9b9bad1103e97832"
    train_filename = "Train.zip"
    test_filename = "Test.zip"
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

    sensor_size = (34, 34)
    ordering = "xytp"

    def __init__(
        self,
        save_to,
        train=True,
        download=True,
        transform=None,
        target_transform=None,
        first_saccade_only=False,
    ):
        super(NMNIST, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.location_on_system = os.path.join(save_to, "nmnist/")
        self.first_saccade_only = first_saccade_only
        self.samples = []
        self.targets = []

        if train:
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "Train"
        else:
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "Test"

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("bin"):
                    self.samples.append(path + "/" + file)
                    label_number = int(path[-1])
                    self.targets.append(label_number)

    def __getitem__(self, index):
        events = self._read_dataset_file(self.samples[index])
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
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
        extract_archive(os.path.join(self.location_on_system, self.train_filename))
        extract_archive(os.path.join(self.location_on_system, self.test_filename))

    def _read_dataset_file(self, filename):
        f = open(filename, "rb")
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = (
            ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        )

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        if self.first_saccade_only:
            td_indices = np.where(all_ts < 100000)[0]

        td = np.empty([td_indices.size, 4], dtype=np.int32)
        td[:, 0] = all_x[td_indices]
        td[:, 1] = all_y[td_indices]
        td[:, 2] = all_ts[td_indices]
        td[:, 3] = all_p[td_indices]

        return td
