import os
import loris
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)
from numpy.lib.recfunctions import structured_to_unstructured


class NCARS(Dataset):
    """N-Cars dataset <https://www.prophesee.ai/dataset-n-cars-download/>. Events have (txyp) ordering.
    ::

        @inproceedings{sironi2018hats,
          title={HATS: Histograms of averaged time surfaces for robust event-based object classification},
          author={Sironi, Amos and Brambilla, Manuele and Bourdis, Nicolas and Lagorce, Xavier and Benosman, Ryad},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1731--1740},
          year={2018}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or verify existing files. If True and a file with the same
                    name and correct hash is already in the directory, download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
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

    sensor_size = None  # different for every recording
    minimum_y_value = 140
    dtype = np.dtype([(("ts", "t"), "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")])
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
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = loris.read_file(self.samples[index])["events"]
        events.dtype.names = ("t", "x", "y", "p")
        events["y"] -= self.minimum_y_value
        events["y"] = self.sensor_size[1] - 1 - events["y"]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
