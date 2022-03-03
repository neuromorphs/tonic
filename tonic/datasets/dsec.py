import os
import numpy as np
import h5py
from typing import Optional, List, Union, Callable
from tonic.dataset import Dataset
from tonic.download_utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    list_files,
)


class DSEC(Dataset):
    """DSEC dataset <https://dsec.ifi.uzh.ch/>. Events have (xytp) ordering.

    .. note:: To be able to read this dataset, you will need `hdf5plugin` and `PIL` packages installed.

    .. note:: This is a fairly large dataset, so in order to speed up training scripts, Tonic will only
              do some lightweight file verification based on the filenames whenever you instantiate this
              class. If your download gets interrupted and you are left with a corrupted file on disk,
              Tonic will not be able to detect that and just proceed to download files that are not yet
              on disk. If you experience issues loading a particular recording, delete that folder
              manually and Tonic will re-download it the next time.

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset. No ground truth available for test set.
        recording (str, optional): Optional parameter to load a selection of recordings by providing a string or a list
                                   thereof, such as 'interlaken_00_c' or ['thun_00_a', 'zurich_city_00_a']. Cannot mix
                                   across train/test. Defaults to None which downloads all train or test recordings.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://download.ifi.uzh.ch/rpg/DSEC/"

    recordings = {
        "train": [
            "interlaken_00_c",
            "interlaken_00_d",
            "interlaken_00_e",
            "interlaken_00_f",
            "interlaken_00_g",
            "thun_00_a",
            "zurich_city_00_a",
            "zurich_city_00_b",
            "zurich_city_01_a",
            "zurich_city_01_b",
            "zurich_city_01_c",
            "zurich_city_01_d",
            "zurich_city_01_e",
            "zurich_city_01_f",
            "zurich_city_02_a",
            "zurich_city_02_b",
            "zurich_city_02_c",
            "zurich_city_02_d",
            "zurich_city_02_e",
            "zurich_city_03_a",
            "zurich_city_04_a",
            "zurich_city_04_b",
            "zurich_city_04_c",
            "zurich_city_04_d",
            "zurich_city_04_e",
            "zurich_city_04_f",
            "zurich_city_05_a",
            "zurich_city_05_b",
            "zurich_city_06_a",
            "zurich_city_07_a",
            "zurich_city_08_a",
            "zurich_city_09_a",
            "zurich_city_09_b",
            "zurich_city_09_c",
            "zurich_city_09_d",
            "zurich_city_09_e",
            "zurich_city_10_a",
            "zurich_city_10_b",
            "zurich_city_11_a",
            "zurich_city_11_b",
            "zurich_city_11_c",
        ],
        "test": [
            "thun_01_a",
            "thun_01_b",
            "interlaken_00_a",
            "interlaken_00_b",
            "interlaken_01_a",
            "zurich_city_12_a",
            "zurich_city_13_a",
            "zurich_city_13_b",
            "zurich_city_14_a",
            "zurich_city_14_b",
            "zurich_city_14_c",
            "zurich_city_15_a",
        ],
    }

    data_filenames = [
        ["events_left", ".zip"],
        ["events_right", ".zip"],
        ["image_timestamps", ".txt"],
        ["image_exposure_timestamps_left", ".txt"],
        ["image_exposure_timestamps_right", ".txt"],
        ["images_rectified_left", ".zip"],
        ["images_rectified_right", ".zip"],
    ]

    target_filenames = [
        ["disparity_event", ".zip"],
        ["disparity_image", ".zip"],
        ["disparity_timestamps", ".txt"],
    ]

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        recording: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(DSEC, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.train = train
        self.train_or_test = "train" if self.train else "test"

        if recording:
            self.selection = recording
            if not isinstance(self.selection, list):
                self.selection = [self.selection]

            for recording in self.selection:
                if recording not in self.recordings[self.train_or_test]:
                    raise RuntimeError(
                        f"Recording {recording} is not in {self.train_or_test} set."
                    )
        else:
            self.selection = self.recordings[self.train_or_test]

        self.filenames = self.data_filenames
        if self.train:
            # add ground truth files when train=True
            self.filenames += self.target_filenames

        self._check_exists()

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (data, target) where data is another tuple of ((events_left, events_right),
            (image_timestamps, images_left, images_right)) and target is either a tuple of
            (disparity_timestamps, disparity_events, disparity_images) if train=True or None if train=False
        """
        import hdf5plugin  # necessary to read event files
        from PIL import Image  # necessary to read images

        recording = self.selection[index]
        base_folder = os.path.join(self.location_on_system, recording)

        # events
        events_left_file = h5py.File(
            os.path.join(base_folder, "events_left", "events.h5")
        )["events"]
        events_left = np.column_stack(
            (
                events_left_file["x"][()],
                events_left_file["y"][()],
                events_left_file["t"][()],
                events_left_file["p"][()],
            )
        )
        events_left = np.lib.recfunctions.unstructured_to_structured(
            events_left, self.dtype
        )

        events_right_file = h5py.File(
            os.path.join(base_folder, "events_right", "events.h5")
        )["events"]
        events_right = np.column_stack(
            (
                events_right_file["x"][()],
                events_right_file["y"][()],
                events_right_file["t"][()],
                events_right_file["p"][()],
            )
        )
        events_right = np.lib.recfunctions.unstructured_to_structured(
            events_right, self.dtype
        )

        # images
        images_rectified_left_filenames = list_files(
            os.path.join(base_folder, "images_rectified_left"), ".png", prefix=True
        )
        images_left = np.stack(
            [np.array(Image.open(file)) for file in images_rectified_left_filenames]
        )

        images_rectified_right_filenames = list_files(
            os.path.join(base_folder, "images_rectified_right"), ".png", prefix=True
        )
        images_right = np.stack(
            [np.array(Image.open(file)) for file in images_rectified_right_filenames]
        )

        with open(
            os.path.join(
                base_folder, "image_timestamps", f"{recording}_image_timestamps.txt"
            )
        ) as f:
            image_timestamps = np.array([int(line) for line in f.readlines()])
        image_timestamps -= image_timestamps[0]

        data = (
            (events_left, events_right),
            (image_timestamps, images_left, images_right),
        )

        if self.transform is not None:
            data = self.transform(data)

        targets = None
        if self.train:
            # ground truth
            disparity_event_filenames = list_files(
                os.path.join(base_folder, "disparity_event"), ".png", prefix=True
            )
            disparity_events = np.stack(
                [np.array(Image.open(file)) for file in disparity_event_filenames]
            )

            disparity_image_filenames = list_files(
                os.path.join(base_folder, "disparity_image"), ".png", prefix=True
            )
            disparity_images = np.stack(
                [np.array(Image.open(file)) for file in disparity_event_filenames]
            )

            with open(
                os.path.join(
                    base_folder,
                    "disparity_timestamps",
                    f"{recording}_disparity_timestamps.txt",
                )
            ) as f:
                disparity_timestamps = np.array([int(line) for line in f.readlines()])
            disparity_timestamps -= disparity_timestamps[0]

            targets = (disparity_timestamps, disparity_events, disparity_images)

            if self.target_transform is not None:
                targets = self.target_transform(targets)

        return data, targets

    def __len__(self):
        return len(self.selection)

    def _check_exists(self):
        for recording in self.selection:
            scene_url = f"{self.base_url}{self.train_or_test}/{recording}/{recording}_"
            for filename, extension in self.filenames:
                if check_integrity(
                    os.path.join(
                        self.location_on_system,
                        recording,
                        filename,
                        f"{recording}_{filename + extension}",
                    )
                ):
                    continue
                file_folder = os.path.join(self.location_on_system, recording, filename)
                os.makedirs(file_folder, exist_ok=True)
                url = scene_url + filename + extension
                if extension == ".zip":
                    download_and_extract_archive(url, file_folder)
                else:
                    download_url(url, file_folder)
