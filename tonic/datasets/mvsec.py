import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from tonic.dataset import Dataset
from tonic.download_utils import check_integrity, download_url


class MVSEC(Dataset):
    """The Multi Vehicle Stereo Event Camera Dataset <https://daniilidis-group.github.io/mvsec/>. Events have (xytp) ordering.
    ::

        @article{zihao2018multi,
          title={The Multi Vehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception},
          author={Zihao Zhu, Alex and Thakur, Dinesh and Ozaslan, Tolga and Pfrommer, Bernd and Kumar, Vijay and Daniilidis, Kostas},
          journal={arXiv e-prints},
          pages={arXiv--1801},
          year={2018}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        scene (string): Choose one of 4 scenes: outdoor_night, outdoor_day, indoor_flying, motorcycle.
                        If you already have the data on your system, make sure to place the .bag files in a subfolder
                        'MVSEC/{scene}/bag_files.bag'.
        transform (callable, optional): A callable of transforms to apply to events and / or images for both left and right cameras.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    resources = {
        "outdoor_night": [
            ["outdoor_night1_data.bag", "534bea503649eeb2801316704d6ab041"],
            ["outdoor_night1_gt.bag", "7e169e4048307e01f1a7ba5931ca7d4d"],
            ["outdoor_night2_data.bag", "371ff73324ba94ecb368b4c220dc8e54"],
            ["outdoor_night2_gt.bag", "d235d28f1c93203a1d7738f8e7a67ca3"],
            ["outdoor_night3_data.bag", "fc40889a48de7b28e6e2506125b229ac"],
            ["outdoor_night3_gt.bag", "2bbdffc94f8dd54f71486fcecfc82fe6"],
        ],
        "motorcycle": [
            ["motorcycle_data.bag", "ae1d929563c63c4d15e0e3d3412d41c4"],
            ["motorcycle_gt.bag", "bebf1cd58837abd1ca625ba219b47388"],
        ],
        "indoor_flying": [
            ["indoor_flying1_data.bag", "fd01f35eb52a754e8195d478ed2a00a2"],
            ["indoor_flying1_gt.bag", "277b97ad46f9dba3896f651ca47297aa"],
            ["indoor_flying2_data.bag", "015ac75086b248167e9602b72485e0eb"],
            ["indoor_flying2_gt.bag", "3c7ba64cd7bede77b1809cc151e54ed2"],
            ["indoor_flying3_data.bag", "fda3408d3c72b2d7445540e5bdbd6396"],
            ["indoor_flying3_gt.bag", "cae5648c84ec09d0316a8d4805dee62e"],
            ["indoor_flying4_data.bag", "30ba3a744dcd4408fc4102e88c636acc"],
            ["indoor_flying4_gt.bag", "f050c886fb34f3890fcf85680e267a21"],
        ],
        "outdoor_day": [
            ["outdoor_day1_data.bag", "7438c34b71d08ff38f52cef68834e9be"],
            ["outdoor_day1_gt.bag", "36ec3dcd0a222c4c2102641a2dc91ff0"],
            ["outdoor_day2_data.bag", "536d20bc59720b995df49f925f96b74d"],
            ["outdoor_day2_gt.bag", "69fb399411d7098b3e2cf3850f593e7b"],
        ],
    }

    base_url = "http://visiondata.cis.upenn.edu/mvsec/"
    sensor_size = (346, 260, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, scene, transform=None, target_transform=None):
        super(MVSEC, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.scene = scene
        if not scene in self.resources.keys():
            raise RuntimeError(
                f"Scene {scene} is not available or in the wrong format. Select one of: indoor_flying, outdoor_day, outdoor_night, motorcycle."
            )

        if not self._check_exists():
            self.download()

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, targets), where data is another tuple of (events_left, events_right, imu_left,
            imu_right, images_left, images_right) and targets is a tuple of (depth_rect_left,
            depth_rect_right, pose) for ground truths.
        """
        # decode data file
        filename = os.path.join(
            self.location_on_system,
            self.scene,
            self.resources[self.scene][index * 2][0],
        )
        topics = importRosbag(filename, log="ERROR")
        events_left = topics["/davis/left/events"]
        events_left["ts"] -= events_left["ts"][0]
        events_left["ts"] *= 1e6
        events_left = np.column_stack(
            (events_left["x"], events_left["y"], events_left["ts"], events_left["pol"])
        )
        events_left = np.lib.recfunctions.unstructured_to_structured(
            events_left, self.dtype
        )
        events_right = topics["/davis/right/events"]
        events_right["ts"] -= events_right["ts"][0]
        events_right["ts"] *= 1e6
        events_right = np.column_stack(
            (
                events_right["x"],
                events_right["y"],
                events_right["ts"],
                events_right["pol"],
            )
        )
        events_right = np.lib.recfunctions.unstructured_to_structured(
            events_right, self.dtype
        )
        imu_left = topics["/davis/left/imu"]
        imu_right = topics["/davis/right/imu"]
        images_left = topics["/davis/left/image_raw"]
        images_left = np.stack(images_left["frames"])
        images_right = topics["/davis/right/image_raw"]
        images_right = np.stack(images_right["frames"])
        data = events_left, events_right, imu_left, imu_right, images_left, images_right

        # decode ground truth file
        filename = os.path.join(
            self.location_on_system,
            self.scene,
            self.resources[self.scene][index * 2 + 1][0],
        )
        topics = importRosbag(filename, log="ERROR")
        depth_left = topics["/davis/left/depth_image_raw"]
        depth_left = np.stack(depth_left["frames"])
        depth_right = topics["/davis/right/depth_image_raw"]
        depth_right = np.stack(depth_right["frames"])
        depth_rect_left = topics["/davis/left/depth_image_rect"]
        depth_rect_left = np.stack(depth_rect_left["frames"])
        depth_rect_right = topics["/davis/right/depth_image_rect"]
        depth_rect_right = np.stack(depth_rect_right["frames"])
        pose = topics["/davis/left/pose"]
        targets = depth_rect_left, depth_rect_right, pose

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.transform(targets)
        return data, targets

    def __len__(self):
        # divided by two because of data and ground truth file per recording
        return len(self.resources[self.scene]) // 2

    def download(self):
        for (filename, md5_hash) in self.resources[self.scene]:
            download_url(
                url=os.path.join(self.base_url, self.scene, filename),
                root=os.path.join(self.location_on_system, self.scene),
                filename=filename,
                md5=md5_hash,
            )

    def _check_exists(self):
        files_present = list(
            [
                check_integrity(
                    os.path.join(self.location_on_system, self.scene, filename)
                )
                for (filename, md5_hash) in self.resources[self.scene]
            ]
        )
        return all(files_present)
