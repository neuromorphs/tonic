import os
import numpy as np
from .dataset import Dataset


class NMNIST(Dataset):
    """NMNIST <https://www.garrickorchard.com/datasets/n-mnist> data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transform: list of transforms to apply to the data
        download: choose to download data or not
    """

    base_url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/"
    test_zip = base_url + "AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1"
    train_zip = base_url + "AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1"
    test_md5 = "69CA8762B2FE404D9B9BAD1103E97832"
    train_md5 = "20959B8E626244A1B502305A9E6E2031"
    test_filename = "nmnist_test.zip"
    train_filename = "nmnist_train.zip"
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
        self.location_on_system = save_to
        self.first_saccade_only = first_saccade_only

        if train:
            self.url = self.train_zip
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "Train"
        else:
            self.url = self.test_zip
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "Test"

        if download:
            self.download()

        if not self.check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file_path = self.location_on_system + "/" + self.folder_name
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("bin"):
                    events = self._read_dataset_file(path + "/" + file)
                    self.data.append(events)
                    label_number = int(path[-1])
                    self.targets.append(label_number)

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

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
