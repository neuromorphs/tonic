import os
import urllib.request
import numpy as np
from typing import Callable, Optional
from tonic.dataset import Dataset
from dv import LegacyAedatFile

class POKERDVS131(Dataset):
    """
    Original 131-sample Poker-DVS dataset.
    Downloads the 3 AEDAT files (cards_1.aedat, cards_2.aedat, cards_3.aedat),
    segments recordings, assigns labels, and exposes Tonic-style events.

    Classes: cl = club, di = diamond, he = heart, sp = spade.
    """

    base_url = "http://www2.imse-cnm.csic.es/caviar/POKER_DVS/"
    files = ["cards_1.aedat", "cards_2.aedat", "cards_3.aedat"]

    # Label order used in the original paper: cl, he, di, sp
    # We will map symbols by their position in the sequence, following MATLAB scripts.
    classes = ["cl", "he", "di", "sp"]
    int_classes = dict(zip(classes, range(4)))

    sensor_size = (128, 128, 2)  # Poker-DVS sensor native resolution
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])

    # segmentation hyperparameters (taken from extr_symb.m)
    idle_threshold_us = 20000  # pauses > 20ms indicate a new recording

    def __init__(
        self,
        save_to: str,
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

        self.location = save_to
        os.makedirs(self.location, exist_ok=True)

        # 1) ensure files exist
        self._download_all()

        # 2) load, segment, label
        self.data = []
        self.targets = []

        for f in self.files:
            path = os.path.join(self.location, f)
            #events = tonic.io.read_aedat4(path)
            with LegacyAedatFile(path) as f_aedat:
                events = np.array(
                    [
                        (e.timestamp, e.x, e.y, e.polarity)
                        for e in f_aedat._read_event_v2()
                    ],
                    dtype=self.dtype,
                )

            self._segment_events(events)

    # -----------------------------------------------------------
    # segmentation logic (equivalent to extr_symb.m)
    # -----------------------------------------------------------
    def _segment_events(self, events):
        t = events["t"]

        # find long pauses that separate consecutive samples
        dt = np.diff(t)
        split_indices = np.where(dt > self.idle_threshold_us)[0]

        # segment start and end indices
        segments = []
        start = 0
        for idx in split_indices:
            segments.append((start, idx))
            start = idx + 1
        segments.append((start, len(events) - 1))

        # assign label in rotating order cl → he → di → sp
        # order in original dataset by recording session
        for i, (s, e) in enumerate(segments):
            sample = events[s:e+1]

            label_idx = i % 4
            label_str = self.classes[label_idx]

            self.data.append(sample)
            self.targets.append(self.int_classes[label_str])

    # -----------------------------------------------------------
    # downloading utilities
    # -----------------------------------------------------------
    def _download_all(self):
        for f in self.files:
            dest = os.path.join(self.location, f)
            if not os.path.isfile(dest):
                print(f"Downloading {f} ...")
                urllib.request.urlretrieve(self.base_url + f, dest)

    def __getitem__(self, index):
        events = self.data[index].copy()
        target = self.targets[index]

        if self.transform:
            events = self.transform(events)
        if self.target_transform:
            target = self.target_transform(target)
        if self.transforms:
            events, target = self.transforms(events, target)

        return events, target

    def __len__(self):
        return len(self.data)
