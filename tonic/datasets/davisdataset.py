import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class DAVISDATA(VisionDataset):
    """DAVIS Event Camera Dataset <http://rpg.ifi.uzh.ch/davis_data.html> data set.

    Args:
        save_to (string): Location to save files to on disk. Will save files in a sub folder 'davis_dataset'.
        recording (string): Use the name of the recording or a list thereof to load it, for example 'dynamic_6dof'
                            or ['slider_far', 'urban']. See project homepage for a list of available recordings.
                            Can use 'all' to load every recording.
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to events and/or images.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, imu, images, opti_track_ground_truth).
    """

    base_url = "http://rpg.ifi.uzh.ch/datasets/davis/"
    recordings = {  # recording names and their md5 hash
        "boxes_6dof": "c919cebc25e564935abfc0a3954bf016",
        "boxes_rotation": "1ec1ef4f354ce1908ed091703c1f98d0",
        "boxes_translation": "4fd11a69015022c72b375a4d000ee8cd",
        "calibration": "6e67817c7b6d2e95ca5796e89d219bac",
        "dynamic_6dof": "16e8cb3b151da15ff1fe6019b517e695",
        "dynamic_rotation": "77a2b0a87582e10e702365a38dc0c93c",
        "dynamic_translation": "2db36fd0c2945cbace48735f0416c506",
        "hdr_boxes": "67661ed4b472189432e35e8849faf200",
        "hdr_poster": "b59e37b616a3eda81184183b12b0dce5",
        "office_spiral": "60fe934450c558aff90ff5ba6b370a05",
        "office_zigzag": "3eba8fc8f42adbfd394b630852ce1f78",
        "outdoors_running": "7db2de811cb22b71fa34abd4ab1bba6b",
        "outdoors_walking": "2ef9b03c87d3c565d30211b7dcfaabc5",
        "poster_6dof": "e42ef11f523a52f11921cdb4a0fca231",
        "poster_rotation": "abcd843a894546775e5dda3560979edf",
        "poster_translation": "bb6736c56ff38f07cbe72613b72d25ed",
        "shapes_6dof": "9b9495ace2dd82881bcc5c0620dd595f",
        "shapes_rotation": "ee436cfe74b545fa25a2534f3ef021df",
        "shapes_translation": "2a805ba32671b237e4e13c023f276be9",
        "slider_close": "9426bbb70334c7b849dd5dd38eb7f2a9",
        "slider_depth": "b38a7a373f170f4b6aeca4f36e06f71a",
        "slider_far": "0f341da6aec0fd1801129e3d3a9981fa",
        "slider_hdr_close": "34cc10dd212ca1fddd3a8584046d5d1c",
        "slider_hdr_far": "c310f4f2d62cdf7b8d1c0a49315fb253",
        "urban": "c22db0b3ecbcbba8d282b0d8c3393851",
    }

    sensor_size = (180, 240)
    ordering = "txyp"

    def __init__(
        self, save_to, recording, download=True, transform=None, target_transform=None,
    ):
        super(DAVISDATA, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        folder_name = "davis_dataset"
        self.location_on_system = os.path.join(save_to, folder_name)
        self.selection = (
            list(self.recordings.keys()) if recording == "all" else recording
        )
        if not isinstance(self.selection, list):
            self.selection = [self.selection]

        for recording in self.selection:
            if recording not in self.recordings:
                raise RuntimeError(
                    "Recording {} is not available or in the wrong format.".format(
                        recording
                    )
                )

        if download:
            self.download()
        else:
            for recording in self.selection:
                if not check_integrity(
                    os.path.join(self.location_on_system, recording + ".bag"),
                    self.recordings[recording],
                ):
                    raise RuntimeError(
                        "Recording not found or corrupted."
                        + " You can use download=True to download it"
                    )

    def __getitem__(self, index):
        filename = os.path.join(self.location_on_system, self.selection[index] + ".bag")
        topics = importRosbag(filename, log="ERROR")
        events = topics["/dvs/events"]
        events = np.stack((events["ts"], events["x"], events["y"], events["pol"])).T
        imu = topics["/dvs/imu"]
        images = topics["/dvs/image_raw"]
        images["frames"] = np.stack(images["frames"])
        target = topics["/optitrack/davis"]

        if self.transform is not None:
            events = self.transform(
                events, self.sensor_size, self.ordering, images=images
            )
        return events, imu, images, target

    def __len__(self):
        return len(self.selection)

    def download(self):
        for recording in self.selection:
            download_url(
                self.base_url + recording + ".bag",
                self.location_on_system,
                filename=recording + ".bag",
                md5=self.recordings[recording],
            )
