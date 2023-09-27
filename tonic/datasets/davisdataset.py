import os
from typing import Callable, List, Optional, Union

import numpy as np
from importRosbag.importRosbag import importRosbag

from tonic.dataset import Dataset
from tonic.download_utils import check_integrity, download_url
from tonic.io import make_structured_array


class DAVISDATA(Dataset):
    """`DAVIS event camera dataset <http://rpg.ifi.uzh.ch/davis_data.html>`_
    ::

        @article{mueggler2017event,
          title={The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM},
          author={Mueggler, Elias and Rebecq, Henri and Gallego, Guillermo and Delbruck, Tobi and Scaramuzza, Davide},
          journal={The International Journal of Robotics Research},
          volume={36},
          number={2},
          pages={142--149},
          year={2017},
          publisher={SAGE Publications Sage UK: London, England}
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will save files in a sub folder 'davis_dataset'.
        recording (string): Use the name of the recording or a list thereof to download it, for example 'dynamic_6dof'
                            or ['slider_far', 'urban']. See project homepage for a list of available recordings.
                            Can use 'all' to download all available recordings.
        transform (callable, optional): A callable of transforms to apply to events and/or images.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
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

    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names
    folder_name = ""

    def __init__(
        self,
        save_to: str,
        recording: Union[str, List[str]],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

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

        if not self._check_exists():
            self.download()

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, target), where data is another tuple of (events, imu, images) and target is the opti track ground truth
        """
        filename = os.path.join(self.location_on_system, self.selection[index] + ".bag")
        topics = importRosbag(filename, log="ERROR")
        events = topics["/dvs/events"]
        events["ts"] -= events["ts"][0]
        events["ts"] *= 1e6
        events = make_structured_array(
            events["ts"], events["x"], events["y"], events["pol"], dtype=self.dtype
        )

        if "/dvs/imu" in topics.keys():
            imu = topics["/dvs/imu"]
            imu["ts"] = ((imu["ts"] - imu["ts"][0]) * 1e6).astype(int)
        else:
            imu = None
        images = topics["/dvs/image_raw"]
        images["frames"] = np.stack(images["frames"])
        images["ts"] = ((images["ts"] - images["ts"][0]) * 1e6).astype(int)
        data = (events, imu, images)
        try:
            target = topics["/optitrack/davis"]
            target["ts"] = ((target["ts"] - target["ts"][0]) * 1e6).astype(int)
        except KeyError:
            target = None

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            data, target = self.transforms(data, target)
        return data, target

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

    def _check_exists(self):
        # check if all filenames are correct
        files_present = list(
            [
                check_integrity(
                    os.path.join(self.location_on_system, recording + ".bag")
                )
                for recording in self.selection
            ]
        )
        return all(files_present)
