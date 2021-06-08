import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url


class MVSEC(VisionDataset):
    """The Multi Vehicle Stereo Event Camera Dataset <https://daniilidis-group.github.io/mvsec/>.

    Args:
        save_to (string): Location to save files to on disk.
        scene (string): Choose one of 4 scenes: outdoor_night, outdoor_day, indoor_flying, motorcycle
        download (bool): Choose to download data or verify existing files. If True and a file with the same name is in the directory, 
                        it will be verified and re-download is automatically skipped. If False, existing files will be
                        be verified. If you already have the data on your system, make sure to place it in a subfolder
                        'MVSEC/{scene}', where {scene} is one of the available strings (see parameter above).
        transform (callable, optional): A callable of transforms to apply to events and / or images for both left and right cameras.
        
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a mix of data and ground truth in a tuple of 
        (events_left, events_right, imu_left, imu_right, images_left, images_right, depth_rect_left, depth_rect_right, pose).
    """

    resources = {
        "outdoor_night": [
            ['outdoor_night1_data.bag', '534bea503649eeb2801316704d6ab041'],
            ['outdoor_night1_gt.bag', '7e169e4048307e01f1a7ba5931ca7d4d'],
            ['outdoor_night2_data.bag', '371ff73324ba94ecb368b4c220dc8e54'],
            ['outdoor_night2_gt.bag', 'd235d28f1c93203a1d7738f8e7a67ca3'],
            ['outdoor_night3_data.bag', 'fc40889a48de7b28e6e2506125b229ac'],
            ['outdoor_night3_gt.bag', '2bbdffc94f8dd54f71486fcecfc82fe6'],
        ],
        "motorcycle": [
            ['motorcycle_data.bag', 'ae1d929563c63c4d15e0e3d3412d41c4'],
            ['motorcycle_gt.bag', 'bebf1cd58837abd1ca625ba219b47388'], 
        ],
        "indoor_flying": [
            ['indoor_flying1_data.bag', 'fd01f35eb52a754e8195d478ed2a00a2'],
            ['indoor_flying1_gt.bag', '277b97ad46f9dba3896f651ca47297aa'],
            ['indoor_flying2_data.bag', '015ac75086b248167e9602b72485e0eb'],
            ['indoor_flying2_gt.bag', '3c7ba64cd7bede77b1809cc151e54ed2'],
            ['indoor_flying3_data.bag', 'fda3408d3c72b2d7445540e5bdbd6396'],            
            ['indoor_flying3_gt.bag', 'cae5648c84ec09d0316a8d4805dee62e'],
            ['indoor_flying4_data.bag', '30ba3a744dcd4408fc4102e88c636acc'],
            ['indoor_flying4_gt.bag', 'f050c886fb34f3890fcf85680e267a21'],
        ],
        "outdoor_day": [
            ['outdoor_day1_data.bag', '7438c34b71d08ff38f52cef68834e9be'],
            ['outdoor_day1_gt.bag', '36ec3dcd0a222c4c2102641a2dc91ff0'],
            ['outdoor_day2_data.bag', '536d20bc59720b995df49f925f96b74d'],
            ['outdoor_day2_gt.bag', '69fb399411d7098b3e2cf3850f593e7b'],
        ],
    }

    base_url = "http://visiondata.cis.upenn.edu/mvsec/"
    sensor_size = (346,260)
    ordering = "xytp"

    def __init__(
        self, save_to, scene, download=True, transform=None, target_transform=None,
    ):
        super(MVSEC, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.location_on_system = os.path.join(save_to, "MVSEC")
        self.scene = scene
        if not scene in self.resources.keys():
            raise RuntimeError("Scene {} is not available or in the wrong format. Select one of: indoor_flying, outdoor_day, outdoor_night, motorcycle. ".format(scene))

        if download:
            self.download()
        else:
            print("Checking folder {}".format(os.path.join(self.location_on_system, self.scene)))
            for (filename, md5_hash) in self.resources[self.scene]:
                print("Checking integrity of file {}".format(filename))
                if not check_integrity(
                    os.path.join(self.location_on_system, self.scene, filename),
                    md5_hash,
                ):
                    raise RuntimeError(
                        "File not found or corrupted."
                        + " You can use download=True to download it"
                    )

    def __getitem__(self, index):
        # decode data file
        filename = os.path.join(self.location_on_system, self.scene, self.resources[self.scene][index*2][0])
        topics = importRosbag(filename, log="ERROR")
        events_left = topics["/davis/left/events"]
        events_left = np.stack((events_left["x"], events_left["y"], events_left["ts"]-events_left["ts"][0], events_left["pol"])).T
        events_right = topics["/davis/right/events"]
        events_right = np.stack((events_right["x"], events_right["y"], events_right["ts"]-events_right["ts"][0], events_right["pol"])).T
        imu_left = topics["/davis/left/imu"]
        imu_right = topics["/davis/right/imu"]
        images_left = topics["/davis/left/image_raw"]
        images_left = np.stack(images_left["frames"])
        images_right = topics["/davis/right/image_raw"]
        images_right = np.stack(images_right["frames"])
        
        # decode ground truth file
        filename = os.path.join(self.location_on_system, self.scene, self.resources[self.scene][index*2+1][0])
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

        if self.transform is not None:
            events_left = self.transform(events_left, self.sensor_size, self.ordering, images=images_left)
        if self.transform is not None:
            events_right = self.transform(events_right, self.sensor_size, self.ordering, images=images_right)
        return events_left, events_right, imu_left, imu_right, images_left, images_right, depth_rect_left, depth_rect_right, pose

    def __len__(self):
        return len(self.resources[self.scene]) // 2  # divided by two because of data and ground truth file per recording 

    def download(self):
        for (filename, md5_hash) in self.resources[self.scene]:
            url = os.path.join(self.base_url, self.scene, filename)
            print("Downloading {}...".format(filename))
            download_url(url, os.path.join(self.location_on_system, self.scene), filename=filename, md5=md5_hash)
