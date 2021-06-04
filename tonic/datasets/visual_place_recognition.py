import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class VisualPlaceRecognition(VisionDataset):
    """Event-Based Visual Place Recognition With Ensembles of Temporal Windows <https://zenodo.org/record/4302805> data set.

    arguments:
        save_to (string): Location to save files to on disk.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://zenodo.org/record/4302805/files/"

    recordings = [  # recording names and their md5 hash
        ["dvs_vpr_2020-04-21-17-03-03.bag", "02f5156504d7780ceb4ebb461ab12788"],
        ["dvs_vpr_2020-04-22-17-24-21.bag", "81cdd7628c551e50a474b1b1e0b2890b"],
        ["dvs_vpr_2020-04-24-15-12-03.bag", "9e4a1d60c2f637f605cfbcdbda668347"],
        ["dvs_vpr_2020-04-27-18-13-29.bag", "b37b720c009611be4bda1bc18147e9db"],
        ["dvs_vpr_2020-04-28-09-14-11.bag", "f4acf1547affdd5c1367385721ca8509"],
        ["dvs_vpr_2020-04-29-06-20-23.bag", "25759c42f7cd77010390231bc2910e22"],
    ]

    sensor_size = (180, 240)
    ordering = "txyp"

    def __init__(
        self, save_to, download=True, transform=None, target_transform=None,
    ):
        super(VisualPlaceRecognition, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        folder_name = "visual_place_recognition"
        self.location_on_system = os.path.join(save_to, folder_name)

        if download:
            self.download()
        else:
            for (recording, md5_hash) in self.recordings:
                if not check_integrity(
                    os.path.join(self.location_on_system, recording), md5_hash
                ):
                    raise RuntimeError(
                        "Dataset file not found or corrupted."
                        + " You can use download=True to download it"
                    )

    def __getitem__(self, index):
        file_path = os.path.join(self.location_on_system, self.recordings[index][0])
        topics = importRosbag(filePathOrName=file_path)
        events = topics["/dvs/events"]
        events = np.stack((events["ts"], events["x"], events["y"], events["pol"])).T
        imu = topics["/dvs/imu"]
        images = topics["/dvs/image_raw"]
        images["frames"] = np.stack(images["frames"])

        if self.transform is not None:
            events = self.transform(
                events, self.sensor_size, self.ordering, images=images
            )
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, imu, images

    def __len__(self):
        return len(self.recordings)

    def download(self):
        for (recording, md5_hash) in self.recordings:
            download_url(
                self.base_url + recording,
                self.location_on_system,
                filename=recording,
                md5=md5_hash,
            )
