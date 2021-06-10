import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class VPR(VisionDataset):
    """Event-Based Visual Place Recognition With Ensembles of Temporal Windows <https://zenodo.org/record/4302805> data set.

    arguments:
        save_to (string): Location to save files to on disk.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://zenodo.org/record/4302805/files/"

    recordings = [  # recording names and their md5 hash
        ["dvs_vpr_2020-04-21-17-03-03.bag", "04473f623aec6bda3d7eadfecfc1b2ce"],
        ["dvs_vpr_2020-04-22-17-24-21.bag", "ca6db080a4054196fe65825bce3db351"],
        ["dvs_vpr_2020-04-24-15-12-03.bag", "909569732e323ff04c94379a787f2a69"],
        ["dvs_vpr_2020-04-27-18-13-29.bag", "e80b6c0434690908d855445792d4de3b"],
        ["dvs_vpr_2020-04-28-09-14-11.bag", "7854ede61c0947adb0f072a041dc3bad"],
        ["dvs_vpr_2020-04-29-06-20-23.bag", "d7ccfeb6539f1e7b077ab4fe6f45193c"],
    ]

    sensor_size = (260, 346)
    ordering = "txyp"

    def __init__(self, save_to, download=True, transform=None, target_transform=None):
        super(VPR, self).__init__(
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
