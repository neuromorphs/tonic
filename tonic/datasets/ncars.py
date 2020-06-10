import os
import numpy as np
from .dataset import Dataset
from numpy.lib.recfunctions import structured_to_unstructured
import loris


class NCARS(Dataset):
    """N-Cars <https://www.prophesee.ai/dataset-n-cars-download/> data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transform: list of transforms to apply to the data
        download: choose to download data or not
    """

    # Train: https://www.neuromorphic-vision.com/public/downloads/ibmGestureTrain.tar.gz
    # Test : https://www.neuromorphic-vision.com/public/downloads/ibmGestureTest.tar.gz
    url = "http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.zip"
    filename = "Prophesee_Dataset_n_cars.zip"
    train_file = "n-cars_train.zip"
    test_file = "n-cars_test.zip"
    file_md5 = "553CE464D6E5E617B3C21CE27C19368E"
    train_md5 = "976D126A651B95D81800B05A3093337B"
    test_md5 = "3B5E8E9A5BFFEB95614B8C0A2BA4E511"
    classes = ["background", "car"]

    class_dict = {"background": 0, "cars": 1}

    sensor_size = (304, 240)
    ordering = "txyp"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None
    ):
        super(NCARS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        if download:
            self.download()

        if not self.check_integrity():
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
            self.extract_archive(os.path.join(save_to, target_zip))
            os.rename(source_path, target_path)

        # We will not be loading everything into memory. Instead, we will keep a list of samples into file
        # Could have reused self.data for that purpose as well.
        self.samples = []

        file_path = target_path
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("dat"):
                    self.samples.append(path + "/" + file)
                    self.targets.append(self.class_dict[os.path.basename(path)])

    def __getitem__(self, index):
        events = loris.read_file(self.samples[index])["events"]
        events = np.array(structured_to_unstructured(events))
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.samples)
