import os
import numpy as np
from .dataset import Dataset


class POKERDVS(Dataset):
    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    train_filename = "pips_train.tar.gz"
    test_filename = "pips_test.tar.gz"
    train_url = base_url + train_filename
    test_url = base_url + test_filename
    train_md5 = "412BCFB96826E4FCB290558E8C150AAE"
    test_md5 = "EEF2BF7D0D3DEFAE89A6FA98B07C17AF"

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

        if not self.check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        file_path = self.location_on_system + "/" + self.folder_name
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.data.append(np.load(path + "/" + file))
                    self.targets.append(self.int_classes[path[-2:]])

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)
