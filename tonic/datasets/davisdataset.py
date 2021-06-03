import os
import numpy as np
import h5py
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class DAVISDATA(VisionDataset):
    """DAVIS Event Camera Dataset <http://rpg.ifi.uzh.ch/davis_data.html> data set.

    arguments:
        save_to (string): Location to save files to on disk.
        recording (string): Use the name of the recording to load it. See project homepage for a list of available recordings.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "http://rpg.ifi.uzh.ch/datasets/davis/"

    recordings = {  # recording names and their md5 hash
        "boxes_6dof": "2c508711f444123734b44ff14b36163b",
        "boxes_rotation": "201b3eb45a75a9af9964c4002cd9895a",
        "boxes_translation": "353192ec47e3764404ba544f63756c81",
        "calibration": "93750e844eb737a75e66a587ef58a8d0",
        "dynamic_6dof": "2fa355235f0d7c88a443fee4e0ef58da",
        "dynamic_rotation": "53282071b96d0f03c6d37ee33e3e75cd",
        "dynamic_translation": "f399c30c8df7186ff771fd2aeb767ef1",
        "hdr_boxes": "d9b2d0a74c69072abf42477376ee45c3",
        "hdr_poster": "3e91a58daa8b279f764db4319f5c901c",
        "office_spiral": "f506f4fed9c69e35d13108662a7616bb",
        "office_zigzag": "86547b7696dc50532db9a05a19e76231",
        "outdoors_running": "bdbb80da73239cb91a7ca3267c76b997",
        "outdoors_walking": "7df86a872515e8f4150fdaf16e30f9de",
        "poster_6dof": "39f11372e539c5b9c0bf053aa57fea91",
        "poster_rotation": "effd58b9c693d2442a26b7390fc2b7be",
        "poster_translation": "7b88d8242a544701a70104f2e6f86fad",
        "shapes_6dof": "4fd3779db4df6b0b067e46c2f15f6d01",
        "shapes_rotation": "1f629824940bd79cb8b6bfeb0d086c7e",
        "shapes_translation": "497e502ddf4f4aed8d328136e6cd79ed",
        "slider_close": "280f4f8b83ba537e660aa0048c0b35d8",
        "slider_depth": "13b57c000f82433857ef396ebff0d247",
        "slider_far": "fa8df212935f931488ab7b3c13fc647a",
        "slider_hdr_close": "db563e1579d164367c9c105a0ec1f4e7",
        "slider_hdr_far": "9cc5e0e4d2575949cb3f02227480307e",
        "urban": "5fd331f7f26df339a1525467d6047929",
    }

    sensor_size = (180, 240)
    ordering = "txyp"

    def __init__(
        self, save_to, recording, download=True, transform=None, target_transform=None,
    ):
        super(DAVISDATA, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        folder_name = "event_camera_dataset"
        self.recording = recording
        self.location_on_system = os.path.join(save_to, folder_name, recording)
        self.filename = recording + ".zip"
        self.url = self.base_url + self.filename
        self.file_md5 = self.recordings[recording]

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

    def __getitem__(self, index):
        with open(os.path.join(self.location_on_system, "events.txt"), "r") as file:
            lines = file.readlines()
            events = np.zeros((len(lines), 4))
            for l, line in enumerate(lines):
                events[l, :] = np.array([float(num) for num in line.split()])

        file_path = os.path.join(self.location_on_system, "images/")
        for path, dirs, files in os.walk(file_path):
            files.sort()
            images = np.zeros((len(files), *self.sensor_size))
            for f, file in enumerate(files):
                if file.endswith("png"):
                    images[f, :, :] = np.array(Image.open(file_path + file))

        target = self.recording
        if self.transform is not None:
            events = self.transform(
                events, self.sensor_size, self.ordering, images=images
            )
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, images, target

    def __len__(self):
        return 1

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
