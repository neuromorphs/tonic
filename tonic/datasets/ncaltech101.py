import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class NCALTECH101(VisionDataset):
    """NCALTECH101 <https://www.garrickorchard.com/datasets/n-caltech101> data set.

    Args:
        save_to (string): Location to save files to on disk.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    url = "https://www.dropbox.com/sh/iuv7o3h2gv6g4vd/AADYPdhIBK7g_fPCLKmG6aVpa?dl=1"
    archive_filename = "N-Caltech101-archive.zip"
    archive_md5 = "989af2c704103341d616b748b5daa70c"
    file_md5 = "66201824eabb0239c7ab992480b50ba3"
    filename = "Caltech101.zip"
    folder_name = "Caltech101"

    sensor_size = None  # all recordings are of different size
    ordering = "xytp"

    def __init__(self, save_to, download=True, transform=None, target_transform=None):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.location_on_system = os.path.join(save_to, "ncaltech-101/")
        self.samples = []
        self.targets = []
        self.x_index = ordering.find("x")
        self.y_index = ordering.find("y")

        if download:
            self.download()

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
        events = self._read_dataset_file(self.samples[index])
        target = self.targets[index]
        events[:, self.x_index] -= events[:, self.x_index].min()
        events[:, self.y_index] -= events[:, self.y_index].min()
        if self.transform is not None:
            sensor_size_x = int(events[:, self.x_index].max() + 1)
            sensor_size_y = int(events[:, self.y_index].max() + 1)
            sensor_size = (sensor_size_x, sensor_size_y)
            events = self.transform(events, sensor_size, self.ordering)
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

        td = np.empty([td_indices.size, 4], dtype=np.int32)
        td[:, 0] = all_x[td_indices]
        td[:, 1] = all_y[td_indices]
        td[:, 2] = all_ts[td_indices]
        td[:, 3] = all_p[td_indices]

        return td
