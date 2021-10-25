import os
import numpy as np
import h5py
from tonic.dataset import Dataset
from tonic.download_utils import check_integrity, download_url


class NTIDIGITS(Dataset):
    """N-TIDIGITS <https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M/edit>.
    Events have (txp) ordering.
    ::

        @article{anumula2018feature,
          title={Feature representations for neuromorphic audio spike streams},
          author={Anumula, Jithendar and Neil, Daniel and Delbruck, Tobi and Liu, Shih-Chii},
          journal={Frontiers in neuroscience},
          volume={12},
          pages={23},
          year={2018},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    #     url = "https://www.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"
    base_url = "https://www.neuromorphic-vision.com/public/downloads/"
    filename = "n-tidigits.hdf5.zip"
    url = base_url + filename
    file_md5 = "eb76091fe71dc2fc9d2a2780e8bfb059"
    folder_name = ""

    sensor_size = (64, 1, 1)
    dtype = np.dtype([("t", int), ("x", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(NTIDIGITS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.train = train

        if not self._check_exists():
            self.download()

        self.data_file = h5py.File(
            os.path.join(self.location_on_system, self.filename[:-4]), "r"
        )

    def __getitem__(self, index):
        if self.train:
            target = bytes.decode(self.data_file["train_labels"][index])
            timestamps = np.array(self.data_file["train_timestamps/" + target])
            addresses = np.array(self.data_file["train_addresses/" + target])
        else:
            target = bytes.decode(self.data_file["test_labels"][index])
            timestamps = np.array(self.data_file["test_timestamps/" + target])
            addresses = np.array(self.data_file["test_addresses/" + target])
        # convert timestamps to microseconds
        timestamps *= 10e5
        events = np.column_stack((timestamps, addresses, np.ones(timestamps.shape[0])))
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.train:
            return len(self.data_file["train_labels"])
        else:
            return len(self.data_file["test_labels"])

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            1, ".hdf5"
        )
