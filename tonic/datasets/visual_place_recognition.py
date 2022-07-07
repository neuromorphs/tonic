import os
import numpy as np
from importRosbag.importRosbag import importRosbag
from tonic.io import make_structured_array
from tonic.dataset import Dataset
from tonic.download_utils import check_integrity, download_url


class VPR(Dataset):
    """`Visual Place Recognition <https://zenodo.org/record/4302805>`_

    Event-Based Visual Place Recognition With Ensembles of Temporal Windows.
    Events have (txyp) ordering.

    .. note::  To be able to read this dataset and its GPS files, you will need the `pynmea2` package installed.

    ::

        @article{fischer2020event,
          title={Event-based visual place recognition with ensembles of temporal windows},
          author={Fischer, Tobias and Milford, Michael},
          journal={IEEE Robotics and Automation Letters},
          volume={5},
          number={4},
          pages={6924--6931},
          year={2020},
          publisher={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
    """

    base_url = "https://zenodo.org/record/4302805/files/"
    recordings = [  # recording names and their md5 hash
        [
            ["dvs_vpr_2020-04-21-17-03-03.bag", "04473f623aec6bda3d7eadfecfc1b2ce"],
            ["20200421_170039-sunset1_concat.nmea", "11f0107a4df845fd315e9134fbee5c1e"],
        ],
        [
            ["dvs_vpr_2020-04-22-17-24-21.bag", "ca6db080a4054196fe65825bce3db351"],
            ["20200422_172431-sunset2_concat.nmea", "ff879bf22a9552a6d8500a98cff6c7f9"],
        ],
        [
            ["dvs_vpr_2020-04-24-15-12-03.bag", "909569732e323ff04c94379a787f2a69"],
            ["20200424_151015-daytime_concat.nmea", "867fdf43ef393ac7e8de251c1a5cd585"],
        ],
        [
            ["dvs_vpr_2020-04-27-18-13-29.bag", "e80b6c0434690908d855445792d4de3b"],
            ["20200427_181204-night_concat.nmea", "441e6673e0dfc8746f76cd646c4aba8d"],
        ],
        [
            ["dvs_vpr_2020-04-28-09-14-11.bag", "7854ede61c0947adb0f072a041dc3bad"],
            ["20200428_091154-morning_concat.nmea", "b86af464ceac478711e52ef4271c198c"],
        ],
        [
            ["dvs_vpr_2020-04-29-06-20-23.bag", "d7ccfeb6539f1e7b077ab4fe6f45193c"],
            ["20200429_061912-sunrise_concat.nmea", "ec04cf35c10eb5b519b11297adef024b"],
        ],
    ]

    sensor_size = (346, 260, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(VPR, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if not self._check_exists():
            self.download()

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (data, target) where data is another tuple of (events, imu, images) and target is gps positional data.
        """
        (bag_filename, _), (gps_filename, _) = self.recordings[index]
        file_path = os.path.join(self.location_on_system, bag_filename)
        topics = importRosbag(filePathOrName=file_path, log="ERROR")
        events = topics["/dvs/events"]
        events["ts"] -= events["ts"][0]
        events["ts"] *= 1e6
        events = make_structured_array(
            events["ts"], events["x"], events["y"], events["pol"], dtype=self.dtype
        )
        imu = topics["/dvs/imu"]
        imu["ts"] = ((imu["ts"] - imu["ts"][0]) * 1e6).astype(int)
        images = topics["/dvs/image_raw"]
        incorrect_shape_indices = [
            i
            for i, image in enumerate(images["frames"])
            if not (image.shape[0] == 260 and image.shape[1] == 346)
        ]
        for (
            index
        ) in (
            incorrect_shape_indices
        ):  # fix frame shapes that don't match for some reason
            shape_diff_x = self.sensor_size[0] - images["frames"][index].shape[1]
            shape_diff_y = self.sensor_size[1] - images["frames"][index].shape[0]
            images["frames"][index] = np.pad(
                images["frames"][index], [(0, shape_diff_y), (0, shape_diff_x), (0, 0)]
            )
        images["frames"] = np.stack(images["frames"])  # errors for some recordings
        images["ts"] = ((images["ts"] - images["ts"][0]) * 1e6).astype(int)
        data = events, imu, images

        targets = self.read_gps_file(
            os.path.join(self.location_on_system, gps_filename)
        )

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.recordings)

    def download(self):
        for recording in self.recordings:
            for filename, md5_hash in recording:
                download_url(
                    self.base_url + filename,
                    self.location_on_system,
                    filename=filename,
                    md5=md5_hash,
                )

    def _check_exists(self):
        # check if all filenames are correct
        files_present = list(
            [
                check_integrity(os.path.join(self.location_on_system, filename))
                for recording in self.recordings
                for filename, md5 in recording
            ]
        )
        return all(files_present)

    # code taken from https://github.com/Tobias-Fischer/ensemble-event-vpr/blob/master/read_gps.py
    def read_gps_file(self, nmea_file_path):
        import pynmea2

        nmea_file = open(nmea_file_path, encoding="utf-8")
        latitudes, longitudes, timestamps = [], [], []
        first_timestamp = None
        previous_lat, previous_lon = 0, 0

        for line in nmea_file.readlines():
            try:
                msg = pynmea2.parse(line)
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                if msg.sentence_type not in ["GSV", "VTG", "GSA"]:
                    dist_to_prev = np.linalg.norm(
                        np.array([msg.latitude, msg.longitude])
                        - np.array([previous_lat, previous_lon])
                    )
                    if (
                        msg.latitude != 0
                        and msg.longitude != 0
                        and msg.latitude != previous_lat
                        and msg.longitude != previous_lon
                        and dist_to_prev > 0.0001
                    ):
                        timestamp_diff = (
                            (msg.timestamp.hour - first_timestamp.hour) * 3600
                            + (msg.timestamp.minute - first_timestamp.minute) * 60
                            + (msg.timestamp.second - first_timestamp.second)
                        )
                        latitudes.append(msg.latitude)
                        longitudes.append(msg.longitude)
                        timestamps.append(timestamp_diff)
                        previous_lat, previous_lon = msg.latitude, msg.longitude

            except pynmea2.ParseError as e:
                continue

        return np.array(np.vstack((latitudes, longitudes, timestamps))).T
