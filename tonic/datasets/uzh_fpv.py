import os
from collections.abc import Callable

import numpy as np
from importRosbag.importRosbag import importRosbag

from tonic.dataset import Dataset
from tonic.download_utils import check_integrity, download_url
from tonic.io import make_structured_array


class UZHFPV(Dataset):
    """`UZH-FPV Drone Racing Dataset <https://fpv.ifi.uzh.ch/>`_

    A dataset of aggressive first-person view drone racing sequences captured with a
    mDAVIS 346 event camera. Sequences are recorded both indoors and outdoors, with the
    camera mounted either forward-facing or at 45 degrees downward.

    Events have (txyp) ordering.

    .. note:: To be able to read this dataset, you will need the `importRosbag` package installed.

    ::

        @InProceedings{Delmerico19icra,
          author    = {Jeffrey Delmerico and Titus Cieslewski and Henri Rebecq and
                       Matthias Faessler and Davide Scaramuzza},
          title     = {Are We Ready for Autonomous Drone Racing? The {UZH-FPV} Drone
                       Racing Dataset},
          booktitle = {{IEEE} Int. Conf. Robot. Autom. ({ICRA})},
          year      = 2019
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        recording (string or list): Name of the recording or list of recording names to use.
            For example ``'indoor_forward_3'`` or ``['indoor_forward_3', 'outdoor_forward_1']``.
            Can be set to ``'all'`` to use all available recordings.
            See the `dataset homepage <https://fpv.ifi.uzh.ch/datasets/>`_ for a full list of
            available recordings.
        transform (callable, optional): A callable of transforms to apply to events and/or images.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    base_url = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/"

    # Maps recording name to True if ground truth is available, False otherwise.
    recordings = {
        "indoor_forward_3": True,
        "indoor_forward_5": True,
        "indoor_forward_6": True,
        "indoor_forward_7": True,
        "indoor_forward_8": False,
        "indoor_forward_9": True,
        "indoor_forward_10": True,
        "indoor_forward_11": False,
        "indoor_forward_12": False,
        "indoor_45_1": False,
        "indoor_45_2": True,
        "indoor_45_3": False,
        "indoor_45_4": True,
        "indoor_45_9": True,
        "indoor_45_11": False,
        "indoor_45_12": True,
        "indoor_45_13": True,
        "indoor_45_14": True,
        "indoor_45_16": False,
        "outdoor_forward_1": True,
        "outdoor_forward_2": False,
        "outdoor_forward_3": True,
        "outdoor_forward_5": True,
        "outdoor_forward_6": False,
        "outdoor_forward_9": False,
        "outdoor_forward_10": False,
        "outdoor_45_1": True,
        "outdoor_45_2": False,
    }

    sensor_size = (346, 260, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        recording: str | list[str],
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        if recording == "all":
            self.selection = list(self.recordings.keys())
        elif isinstance(recording, str):
            self.selection = [recording]
        else:
            self.selection = list(recording)

        for rec in self.selection:
            if rec not in self.recordings:
                raise RuntimeError(
                    f"Recording '{rec}' is not available. "
                    f"Please select from: {list(self.recordings.keys())}."
                )

        if not self._check_exists():
            self.download()

    def __getitem__(self, index):
        """
        Returns:
            tuple of (data, target), where data is a tuple of (events, imu, images) and
            target is the ground truth odometry or None if ground truth is not available.
            events is a structured numpy array with dtype (t, x, y, p).
            imu is a dict with keys 'ts' and IMU measurement fields.
            images is a dict with keys 'frames' (stacked numpy array) and 'ts' (timestamps).
            target is a dict with pose information from ground truth, or None.
        """
        rec = self.selection[index]
        has_gt = self.recordings[rec]

        if has_gt:
            filename = os.path.join(
                self.location_on_system, rec + "_davis_with_gt.bag"
            )
        else:
            filename = os.path.join(self.location_on_system, rec + "_davis.bag")

        topics = importRosbag(filename, log="ERROR")

        # Normalize timestamps to start at 0 and convert from seconds to microseconds
        events = topics["/dvs/events"]
        events["ts"] -= events["ts"][0]
        events["ts"] *= 1e6
        events = make_structured_array(
            events["ts"], events["x"], events["y"], events["pol"], dtype=self.dtype
        )

        # Normalize and convert IMU timestamps to microseconds
        imu = topics["/dvs/imu"]
        imu["ts"] = ((imu["ts"] - imu["ts"][0]) * 1e6).astype(int)

        # Normalize and convert image timestamps to microseconds, stack frames
        images = topics["/dvs/image_raw"]
        images["frames"] = np.stack(images["frames"])
        images["ts"] = ((images["ts"] - images["ts"][0]) * 1e6).astype(int)

        data = events, imu, images

        if has_gt and "/groundtruth/odometry" in topics:
            # Normalize and convert ground truth timestamps to microseconds
            target = topics["/groundtruth/odometry"]
            target["ts"] = ((target["ts"] - target["ts"][0]) * 1e6).astype(int)
        else:
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
        for rec in self.selection:
            has_gt = self.recordings[rec]
            if has_gt:
                filename = rec + "_davis_with_gt.bag"
            else:
                filename = rec + "_davis.bag"
            download_url(
                self.base_url + filename,
                self.location_on_system,
                filename=filename,
            )

    def _check_exists(self):
        files_present = []
        for rec in self.selection:
            has_gt = self.recordings[rec]
            if has_gt:
                filename = rec + "_davis_with_gt.bag"
            else:
                filename = rec + "_davis.bag"
            files_present.append(
                check_integrity(os.path.join(self.location_on_system, filename))
            )
        return all(files_present)
