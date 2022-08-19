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
from tonic.io import make_structured_array


class DSEC(Dataset):
    """`DSEC <https://dsec.ifi.uzh.ch/>`_

    .. note:: To be able to read this dataset, you will need `hdf5plugin` and `PIL` packages installed.

    .. note:: This is a fairly large dataset, so in order to speed up training scripts, Tonic will only
              do some lightweight file verification based on the filenames whenever you instantiate this
              class. If your download gets interrupted and you are left with a corrupted file on disk,
              Tonic will not be able to detect that and just proceed to download files that are not yet
              on disk. If you experience issues loading a particular recording, delete that folder
              manually and Tonic will re-download it the next time.

    Parameters:
        save_to (str): Location to save files to on disk.
        split (str): Can be 'train', 'test' or a selection of individual recordings such as 'interlaken_00_c'
                     or ['thun_00_a', 'zurich_city_00_a']. Cannot mix across train/test.
        data_selection (str): Select which data to load per sample. Can be 'events_left', 'events_right',
                             'images_rectified_left', 'images_rectified_right', 'image_timestamps' or
                             any combination thereof in a list.
        target_selection (str, optional): Select which targets to load. Omitted if split contains training
                                          samples. Can be 'disparity_events', 'disparity_images', 'disparity_timestamps',
                                          'optical_flow' or a combination thereof in a list.
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

    # that's a combination of the different data available, their
    # extension when downloaded and their extension when extracted
    data_names = {
        "events_left": [".zip", ".h5"],
        "events_right": [".zip", ".h5"],
        "image_timestamps": [".txt", ".txt"],
        "image_exposure_timestamps_left": [".txt", ".txt"],
        "image_exposure_timestamps_right": [".txt", ".txt"],
        "images_rectified_left": [".zip", ".png"],
        "images_rectified_right": [".zip", ".png"],
    }

    target_names = {
        "disparity_event": [".zip", ".png"],
        "disparity_image": [".zip", ".png"],
        "disparity_timestamps": [".txt", ".txt"],
        "optical_flow_forward_event": [".zip", ".png"],
        "optical_flow_forward_timestamps": [".txt", ".txt"],
        "optical_flow_backward_event": [".zip", ".png"],
        "optical_flow_backward_timestamps": [".txt", ".txt"],
    }

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: Union[str, List[str]],
        data_selection: Union[str, List[str]],
        target_selection: Optional[Union[str, List[str]]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(DSEC, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if split in ["train", "test"]:
            self.recording_selection = self.recordings[split]
            self.train_or_test = split

        else:
            if not isinstance(split, list):
                split = [split]

            for recording in split:
                if (
                    recording not in self.recordings["train"]
                    and recording not in self.recordings["test"]
                ):
                    raise RuntimeError(
                        f"Recording {recording} is neither in train nor in test set."
                    )
            self.recording_selection = split
            if all([recording in self.recordings["train"] for recording in split]):
                self.train_or_test = "train"
            elif all([recording in self.recordings["test"] for recording in split]):
                self.train_or_test = "test"
            else:
                raise RuntimeError("Cannot mix across train/test split.")
        self.train = self.train_or_test == "train"

        if isinstance(data_selection, str):
            data_selection = [data_selection]

        for data_piece in data_selection:
            if data_piece not in self.data_names.keys():
                raise RuntimeError(
                    f"Selection {data_piece} not available. Please select from the following options: {self.data_names.keys()}."
                )
        self.data_selection = data_selection
        self.selection = data_selection

        if self.train:
            if isinstance(target_selection, str) and target_selection is not None:
                target_selection = [target_selection]
            else:
                target_selection = []
            for data_piece in target_selection:
                if data_piece not in self.target_names.keys():
                    raise RuntimeError(
                        f"Selection {data_piece} not available. Please select from the following options: {self.target_names.keys()}."
                    )
            self.target_selection = target_selection
            self.selection += target_selection
        else:
            if target_selection is not None or len(target_selection) > 0:
                raise Exception("No targets for test set available.")

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

        recording = self.recording_selection[index]
        base_folder = os.path.join(self.location_on_system, recording)

        # events
        events_left_file = h5py.File(
            os.path.join(base_folder, "events_left", "events.h5")
        )["events"]
        events_left = make_structured_array(
            events_left_file["x"][()],
            events_left_file["y"][()],
            events_left_file["t"][()],
            events_left_file["p"][()],
            dtype=self.dtype,
        )

        events_right_file = h5py.File(
            os.path.join(base_folder, "events_right", "events.h5")
        )["events"]
        events_right = make_structured_array(
            events_right_file["x"][()],
            events_right_file["y"][()],
            events_right_file["t"][()],
            events_right_file["p"][()],
            dtype=self.dtype,
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
                [np.array(Image.open(file)) for file in disparity_image_filenames]
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
        return len(self.recording_selection)

    def _check_exists(self):
        all_names = {**self.data_names, **self.target_names}
        for recording in self.recording_selection:
            for data_name in self.selection:
                file_folder = os.path.join(
                    self.location_on_system, recording, data_name
                )
                os.makedirs(file_folder, exist_ok=True)
                extension, extracted_file_extension = all_names[data_name]
                file_name = f"{recording}_{data_name + extension}"

                if any(
                    file.endswith(extracted_file_extension)
                    for file in os.listdir(file_folder)
                ):
                    continue

                url = f"{self.base_url}{self.train_or_test}/{recording}/{file_name}"
                if extension == ".zip":
                    download_and_extract_archive(url, file_folder)
                    if "images" in data_name or "events" in data_name:
                        os.remove(os.path.join(file_folder, file_name))
                else:
                    download_url(url, file_folder)
