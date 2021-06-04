import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)
from numpy.lib.recfunctions import structured_to_unstructured
import loris


class NCARS(VisionDataset):
    """N-Cars <https://www.prophesee.ai/dataset-n-cars-download/> data set.

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    url = "http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.zip"
    filename = "Prophesee_Dataset_n_cars.zip"
    train_file = "n-cars_train.zip"
    test_file = "n-cars_test.zip"
    file_md5 = "553ce464d6e5e617b3c21ce27c19368e"
    train_md5 = "976d126a651b95d81800b05a3093337b"
    test_md5 = "3b5e8e9a5bffeb95614b8c0a2ba4e511"
    classes = ["background", "car"]

    class_dict = {"background": 0, "cars": 1}

    sensor_size = (120, 100)
    minimum_y_value = 140
    ordering = "txyp"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None
    ):
        super(NCARS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.location_on_system = save_to
        self.samples = []
        self.targets = []

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if train:
            target_zip = self.train_file
            source_path = os.path.join(save_to, "train")
            target_path = os.path.join(save_to, "ncars-train")
        else:
            target_zip = self.test_file
            source_path = os.path.join(save_to, "test")
            target_path = os.path.join(save_to, "ncars-test")

        if not os.path.exists(target_path):
            extract_archive(os.path.join(save_to, target_zip))
            os.rename(source_path, target_path)

        file_path = target_path
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("dat"):
                    self.samples.append(path + "/" + file)
                    self.targets.append(self.class_dict[os.path.basename(path)])

    def __getitem__(self, index):
        events = loris.read_file(self.samples[index])["events"]
        events = np.array(structured_to_unstructured(events, dtype=np.float))
        events[:, 2] -= self.minimum_y_value
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
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
