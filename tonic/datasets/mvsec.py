import os
import numpy as np
import h5py
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class MVSEC(VisionDataset):
    """The Multi Vehicle Stereo Event Camera Dataset <https://daniilidis-group.github.io/mvsec/>.

    arguments:
        save_to (string): Location to save files to on disk.
        scene (string): Choose one of 4 scenes: outdoor_night, outdoor_day, indoor_flying
        download (bool): Choose to download data or not. If True and a file with the same name is in the directory, it will be verified and re-download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    resources = {
        "outdoor_night": [
            [
                "outdoor_night1_data.hdf5",
                "1z8b00gWoZnGuzAOSD49KFaX03q1UuKxc/view",
                "placeholder",
            ],
            [
                "outdoor_night1_gt.hdf5",
                "139dZNXHNUtSul0ZLmPu6N39gbvciQZme/view",
                "placeholder",
            ],
        ],
        "outdoor_day": [
            [
                "outdoor_day1_data.hdf5",
                "1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0/view",
                "199141417e41dd71ff988f61d70c5ef9",
            ],
            [
                "outdoor_day1_gt.hdf5",
                "1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e/view",
                "199141417e41dd71ff988f61d70c5ef9",
            ],
            [
                "outdoor_day2_data.hdf5",
                "1fu9GhjYcET00mMN-YbAp3eBK1YMCd3Ox/view",
                "placeholder",
            ],
            [
                "outdoor_day2_gt.hdf5",
                "1zWOA92-Bw4xz1y5CzIROXWFymTFFwBBH/view",
                "placeholder",
            ],
        ],
        "indoor_flying": [
            [
                "indoor_flying4_data.hdf5",
                "1bngs9QbVX3KtESrPLyXJbGyxuZI8j9tb/view",
                "b4021c98b0ddbb13319772ca90a12407",
            ],
            [
                "indoor_flying4_gt.hdf5",
                "1UweN8PPaNlG27qp-ORc89gFRIzaid38o/view",
                "8743240cefbf4e815e1e04daf7ce659a",
            ],
        ],
    }

    base_url = "https://drive.google.com/file/d/"
    sensor_size = (64,)
    ordering = "xytp"

    def __init__(
        self, save_to, scene, download=True, transform=None, target_transform=None,
    ):
        super(MVSEC, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.location_on_system = os.path.join(save_to, "MVSEC")
        self.scene = scene
        assert scene in self.resources.keys()

        if download:
            self.download()

    #         if not check_integrity(
    #             os.path.join(self.location_on_system, self.filename), self.file_md5
    #         ):
    #             raise RuntimeError(
    #                 "Dataset not found or corrupted."
    #                 + " You can use download=True to download it"
    #             )

    def __getitem__(self, index):
        file = h5py.File(os.path.join(self.location_on_system, self.filename), "r")
        events = (
            np.array(file["/davis/left/events"]),
            np.array(file["/davis/right/events"]),
        )
        images = (
            np.array(file["/davis/left/image_raw"]),
            np.array(file["/davis/right/image_raw"]),
        )

        if self.transform is not None:
            events = self.transform(events, self.sensor_size, self.ordering)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, images, target

    def __len__(self):
        return (
            len(self.resources[self.scene]) // 2
        )  # divided by 2 for data and ground truth

    def download(self):
        for (filename, file_id, md5_hash) in self.resources[self.scene]:
            url = self.base_url + file_id
            print("Downloading {}...".format(filename))
            download_url(url, self.location_on_system, filename=filename, md5=md5_hash)
