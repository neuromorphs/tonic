#!/user/bin/env python

import numpy as np
import h5py
import os
from typing import Callable, Optional

from tonic.dataset import Dataset
from tonic.io import make_structured_array
import requests
from tqdm import tqdm

class NTIDIGITS18(Dataset):
    """`N-TIDIGITS18 Dataset <https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit?tab=t.0#heading=h.sbnu5gtazqjq/>`_
    Cochlea Spike Dataset.
    ::

        @article{anumula2018feature,
          title={Feature representations for neuromorphic audio spike streams},
          author={Anumula, Jithendar and Neil, Daniel and Delbruck, Tobi and Liu, Shih-Chii},
          journal={Frontiers in neuroscience},
          volume={12},
          pages={23},
          year={2018},
          publisher={Frontiers Media SA}
        }

    Parameters:
        save_to (string): Location to save files to on disk. Will put files in an 'hsd' subfolder.
        train (bool): If True, uses training subset, otherwise testing subset.
        single_digits (bool): If True, only returns samples with single digits (o, 1, 2, 3, 4, 5, 6, 7, 8, 9, z), with class 0 for 'o' and 11 for 'z'.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    base_url = "https://www.dropbox.com/scl/fi/1x4lxt9yyw25sc3tez8oi/n-tidigits.hdf5?e=2&rlkey=w8gi5udvib2zqzosusa5tr3wq&dl=1"
    filename = "n-tidigits.hdf5"
    file_md5 = "360a2d11e5429555c9197381cf6b58e0"
    folder_name = ""

    sensor_size = (64, 1, 1)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    class_map = {"o": 0,
                 "1": 1,
                 "2": 2,
                 "3": 3,
                 "4": 4,
                 "5": 5,
                 "6": 6,
                 "7": 7,
                 "8": 8,
                 "9": 9,
                 "z": 10}

    def __init__(
            self,
            save_to: str,
            train: bool = True,
            single_digits=False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
        )

        self.url = self.base_url
        # load the data
        self.file_path = os.path.join(self.location_on_system, self.filename)

        if not self._check_exists():
            self.download()

        self.data = h5py.File(self.file_path, 'r')
        self.partition = "train" if train else "test"
        self.single_indices = [i for i in range(len(self.data[f"{self.partition}_labels"])) if
                               len(self.data[f"{self.partition}_labels"][i].decode().split("-")[-1]) == 1]
        self._samples = [x.decode() for x in self.data[f"{self.partition}_labels"]]
        self.single_digits = single_digits

        if single_digits:
            self._samples = [self._samples[i] for i in self.single_indices]

        self.labels = [x.decode().split("-")[-1] for x in self.data[f"{self.partition}_labels"]]

    def download(self) -> None:
        response = requests.get(self.base_url, stream=True)
        if response.status_code == 200:
            print("Downloading N-TIDIGITS from Dropbox at {}...".format(self.base_url))
            file_size = int(response.headers.get('Content-Length', 0))  # get total file size in bytes
            chunk_size = 8192

            os.makedirs(self.location_on_system, exist_ok=True)
            # Initialize progress bar
            with open(os.path.join(self.location_on_system, self.filename), 'wb') as f, tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading",
                    ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            print("Failed to download N-TIDIGITS from Dropbox. Please try again later.")
            response.raise_for_status()
    def __getitem__(self, index):
        sample_id = self._samples[index]
        x = np.asarray(self.data[f"{self.partition}_addresses"][sample_id])
        t = np.asarray(self.data[f"{self.partition}_timestamps"][sample_id])
        events = make_structured_array(
            t * 1e6,
            x,
            1,
            dtype=self.dtype,
        )

        target = sample_id.split("-")[-1]

        if self.single_digits:
            assert len(target) == 1, "Single digit samples requested, but target is not single digit."
            target = self.class_map[target]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self._samples)

    def _check_exists(self):
        return (
            self._is_file_present()
        )