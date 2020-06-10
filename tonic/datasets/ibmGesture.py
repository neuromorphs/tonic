import os
import numpy as np
from .dataset import Dataset


class IBMGesture(Dataset):
    """IBMGesture <http://research.ibm.com/dvsgesture/> data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transform: list of transforms to apply to the data
        download: choose to download data or not
    """

    # Train: https://www.neuromorphic-vision.com/public/downloads/ibmGestureTrain.tar.gz
    # Test : https://www.neuromorphic-vision.com/public/downloads/ibmGestureTest.tar.gz
    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    test_zip = base_url + "ibmGestureTest.tar.gz"
    train_zip = base_url + "ibmGestureTrain.tar.gz"
    test_md5 = "56070E45DADAA85FFF82E0FBFBC06DE5"
    train_md5 = "3A8F0D4120A166BAC7591F77409CB105"
    test_filename = "ibmGestureTest.tar.gz"
    train_filename = "ibmGestureTrain.tar.gz"
    classes = [
        "hand_clapping",
        "right_hand_wave",
        "left_hand_wave",
        "right_arm_clockwise",
        "right_arm_counter_clockwise",
        "left_arm_clockwise",
        "left_arm_counter_clockwise",
        "arm_roll",
        "air_drums",
        "air_guitar",
        "other_gestures",
    ]

    sensor_size = (128, 128)
    ordering = "xypt"

    def __init__(
        self, save_to, train=True, download=True, transform=None, target_transform=None
    ):
        super(IBMGesture, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        # We will not be loading everything into memory. Instead, we will keep a list of samples into file
        # Could have reused self.data for that purpose as well.
        self.samples = []

        self.train = train
        self.location_on_system = save_to

        if train:
            self.url = self.train_zip
            self.file_md5 = self.train_md5
            self.filename = self.train_filename
            self.folder_name = "ibmGestureTrain"
        else:
            self.url = self.test_zip
            self.file_md5 = self.test_md5
            self.filename = self.test_filename
            self.folder_name = "ibmGestureTest"

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
                    self.samples.append(path + "/" + file)
                    self.targets.append(int(file[:-4]))

    def __getitem__(self, index):
        events = np.load(self.samples[index])
        events[:, 3] *= 1000  # convert from ms to us
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.samples)
