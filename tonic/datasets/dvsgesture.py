import os
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class DVSGesture(VisionDataset):
    """DVSGesture <http://research.ibm.com/dvsgesture/> data set.

    Args:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    # Train: https://www.neuromorphic-vision.com/public/downloads/ibmGestureTrain.tar.gz
    # Test : https://www.neuromorphic-vision.com/public/downloads/ibmGestureTest.tar.gz
    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    test_zip = base_url + "ibmGestureTest.tar.gz"
    train_zip = base_url + "ibmGestureTrain.tar.gz"
    test_md5 = "56070e45dadaa85fff82e0fbfbc06de5"
    train_md5 = "3a8f0d4120a166bac7591f77409cb105"
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
        super(DVSGesture, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train
        self.location_on_system = save_to
        self.data = []
        self.samples = []
        self.targets = []

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

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
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

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )

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
