import os
import loris
import numpy
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class NavGesture(VisionDataset):
    """NavGesture <https://www.neuromorphic-vision.com/public/downloads/navgesture/> data set

    Args:
        save_to (string): Location to save files to on disk.
        walk_subset (bool): Choose either NavGesture-sit (default) or NavGesture-walk dataset. No train/test split provided.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    base_url = "https://www.neuromorphic-vision.com/public/downloads/navgesture/"
    sit_filename = "navgesture-sit.zip"
    walk_filename = "navgesture-walk.zip"
    sit_url = base_url + sit_filename
    walk_url = base_url + walk_filename
    sit_md5 = "1571753ace4d9e0946e6503313712c22"
    walk_md5 = "5d305266f13005401959e819abe206f0"

    classes = ["swipe down", "swipe up", "swipe left", "swipe right", "select", "home"]
    class_codes = ["do", "up", "le", "ri", "se", "ho"]
    int_classes = dict(zip(class_codes, range(len(class_codes))))
    sensor_size = (304, 240)
    ordering = "txyp"

    def __init__(
        self,
        save_to,
        walk_subset=False,
        download=True,
        transform=None,
        target_transform=None,
    ):
        super(NavGesture, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.walk_subset = walk_subset
        self.location_on_system = save_to
        self.data = []
        self.targets = []

        if walk_subset:
            self.url = self.walk_url
            self.file_md5 = self.walk_md5
            self.filename = self.walk_filename
            self.folder_name = "navgesture-walk"
        else:
            self.url = self.sit_url
            self.file_md5 = self.sit_md5
            self.filename = self.sit_filename
            self.folder_name = "navgesture-sit"

        self.location_on_system = os.path.join(
            self.location_on_system, self.folder_name
        )
        if not os.path.exists(self.location_on_system):
            os.mkdir(self.location_on_system)

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.samples = []
        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("dat"):
                    self.samples.append(path + "/" + file)
                    self.targets.append(self.int_classes[file[7:9]])

    def __getitem__(self, index):
        events, target = loris.read_file(self.samples[index]), self.targets[index]
        events = numpy.lib.recfunctions.structured_to_unstructured(
            events["events"], dtype=numpy.float
        )
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
        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            for file in files:
                if file.startswith("user") and file.endswith("zip"):
                    extract_archive(os.path.join(self.location_on_system, file))
