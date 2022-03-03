import os
import numpy as np
import h5py
from numpy.lib import recfunctions
from importRosbag.importRosbag import importRosbag
from tonic.dataset import Dataset
from tonic.download_utils import (
    check_integrity,
    download_url,
    download_and_extract_archive,
    list_files,
)
from typing import Union, List, Callable, Optional


class TUMVIE(Dataset):
    """Visual-Inertial Event Dataset <https://vision.in.tum.de/data/datasets/visual-inertial-event-dataset>.
    TUM-VIE is an event camera dataset for developing 3D perception and navigation algorithms. It contains
    handheld and head-mounted sequences in indoor and outdoor environments with rapid motion during sports
    and high dynamic range. TUM-VIE includes challenging sequences where state-of-the art VIO fails or
    results in large drift. Hence, it can help to push the boundary on event-based visual-inertial algorithms.

    The dataset contains:

    * Stereo event data Prophesee Gen4 HD (1280x720 pixels)
    * Stereo grayscale frames at 20Hz (1024x1024 pixels)
    * IMU data at 200Hz
    * 6dof motion capture data at 120Hz (beginning and end of each sequence)

    Timestamps between all sensors are synchronized in hardware.

    .. note:: To be able to read this dataset, you will need `hdf5plugin` and `PIL` packages installed.

    .. note:: Use `bike-night` instead of `bike-dark` in the list of recordings if you want that specific one.

    ::

        @string{iros="International Conference on Intelligent Robots and Systems (IROS)"}
        @inproceedings{klenk2021tumvie,
         author = {S Klenk and J Chui and N Demmel and D Cremers},
         title = {TUM-VIE: The TUM Stereo Visual-Inertial Event Dataset},
         eprint = {2108.07329},
         eprinttype = {arXiv},
         eprintclass = {cs.CV},
         booktitle = {International Conference on Intelligent Robots and Systems (IROS)},
         year = {2021},
         keywords = {tumvie, event camera, dynamic vision sensor, SLAM, vslam},
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will save files in a sub folder 'davis_dataset'.
        recording (string): Use the name of the recording or a list thereof to download it, for example 'skate-hard'
                            or ['skate-hard', 'bike-easy']. See project homepage for a list of available recordings.
                            Can use 'all' to download all available recordings.
        transform (callable, optional): A callable of transforms to apply to events and/or images.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://tumevent-vi.vision.in.tum.de/"
    recordings = [
        "mocap-1d-trans",
        "mocap-3d-trans",
        "mocap-6dof",
        "mocap-desk",
        "mocap-desk2",
        "mocap-shake",
        "mocap-shake2",
        "office-maze",
        "running-easy",
        "running-hard",
        "skate-easy",
        "skate-hard",
        "loop-floor0",
        "loop-floor1",
        "loop-floor2",
        "loop-floor3",
        "floor2-dark",
        "slide",
        "bike-easy",
        "bike-hard",
        "bike-night",
    ]
    filenames = ["events_left.h5", "events_right.h5", "vi_gt_data.tar.gz"]

    sensor_size = (1280, 720, 2)
    dtype = np.dtype([("p", int), ("t", int), ("x", int), ("y", int)])
    ordering = dtype.names
    folder_name = ""

    def __init__(
        self,
        save_to: str,
        recording: Union[str, List[str]],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(TUMVIE, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if recording == "all" or ["all"]:
            self.selection = self.recordings
        else:
            self.selection = recording if type(recording) == list else [recording]

            for recording in self.selection:
                if recording not in self.recordings:
                    raise RuntimeError(f"Recording {recording} is not available.")

        self._check_exists()

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, target), where data is a dictionary of (events_left, events_right, imu)
            and targets is a dictionary of (images_left, images_right, mocap).
        """
        base_folder = os.path.join(self.location_on_system, self.selection[index])

        import hdf5plugin  # necessary to read event files
        from PIL import Image  # necessary to read images

        events_left_file = h5py.File(
            os.path.join(base_folder, self.selection[index] + "-events_left.h5")
        )["events"]
        events_left = np.column_stack(
            (
                events_left_file["p"][()],
                events_left_file["t"][()],
                events_left_file["x"][()],
                events_left_file["y"][()],
            )
        )
        events_left = np.lib.recfunctions.unstructured_to_structured(
            events_left, self.dtype
        )

        events_right_file = h5py.File(
            os.path.join(base_folder, self.selection[index] + "-events_right.h5")
        )["events"]
        events_right = np.column_stack(
            (
                events_right_file["p"][()],
                events_right_file["t"][()],
                events_right_file["x"][()],
                events_right_file["y"][()],
            )
        )
        events_right = np.lib.recfunctions.unstructured_to_structured(
            events_right, self.dtype
        )

        imu_data = []
        with open(os.path.join(base_folder, "imu_data.txt")) as f:
            header = f.readline()
            for line in f.readlines():
                imu_data.append([float(num) for num in line.split()])
        imu_data = np.array(imu_data)

        mocap_data = []
        with open(os.path.join(base_folder, "mocap_data.txt")) as f:
            header = f.readline()
            for line in f.readlines():
                mocap_data.append([float(num) for num in line.split()])
        mocap_data = np.array(mocap_data)

        # images
        images_left_filenames = list_files(
            os.path.join(base_folder, "left_images"), ".jpg", prefix=True
        )
        images_left = np.stack(
            [np.array(Image.open(file)) for file in images_left_filenames]
        )
        with open(
            os.path.join(base_folder, "left_images", "image_timestamps_left.txt")
        ) as f:
            images_left_timestamps = np.array(
                [float(line) for line in f.readlines()[1:]]
            )
        images_left_timestamps -= images_left_timestamps[0]

        images_right_filenames = list_files(
            os.path.join(base_folder, "right_images"), ".jpg", prefix=True
        )
        images_right = np.stack(
            [np.array(Image.open(file)) for file in images_right_filenames]
        )
        with open(
            os.path.join(base_folder, "right_images", "image_timestamps_right.txt")
        ) as f:
            images_right_timestamps = np.array(
                [float(line) for line in f.readlines()[1:]]
            )
        images_right_timestamps -= images_right_timestamps[0]

        data = {
            "events_left": events_left,
            "events_right": events_right,
            "imu": imu_data,
        }
        targets = {
            "images_left": {"frames": images_left, "ts": images_left_timestamps},
            "images_right": {"frames": images_right, "ts": images_right_timestamps},
            "mocap": mocap_data,
        }

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.selection)

    def _check_exists(self):
        for recording in self.selection:
            file_folder = os.path.join(self.location_on_system, recording)
            os.makedirs(file_folder, exist_ok=True)
            for filename in self.filenames:
                if check_integrity(
                    os.path.join(file_folder, f"{recording}-{filename}")
                ):
                    continue
                url = f"{self.base_url}{recording}/{recording}-{filename}"
                if filename.endswith("tar.gz"):
                    download_and_extract_archive(url, file_folder)
                else:
                    download_url(url, file_folder)
