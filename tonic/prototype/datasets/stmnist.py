"""
Novel neuromorphic Spiking Tactile MNIST (ST-MNIST) dataset, which comprises handwritten digits obtained by human participants writing on a neuromorphic tactile sensor array.
The original paper can be found at https://arxiv.org/abs/2005.04319.
Data is provided with the MAT format. 
Download of the compressed dataset has to be done by the user by accessing https://scholarbank.nus.edu.sg/bitstream/10635/168106/2/STMNIST%20dataset%20NUS%20Tee%20Research%20Group.zip, where a form has to be compiled. Then, the path to the ZIP archive has to be provided to the stmnist() function root argument.
"""

from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    Zipper,
    UnZipper, 
    ZipArchiveLoader,
    IterDataPipe
)
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from tonic.download_utils import check_integrity
from typing import Optional, Callable

#####
# Dataset properties.
#####

sensor_size = (10, 10, 2)
dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
MD5 = "2eef16be7356bc1a8f540bb3698c4e0e"

#####
# Functions
#####


def _is_mat_file(dp):
    return dp[0].endswith("mat") and "LUT" not in dp[0]


def _read_label_from_filepath(filepath):
    return int(filepath.split("/")[-2])


def _spiketrain_to_array(matfile):
    # Transposing since the order is (address, event),
    # but we like (event, address).
    mat = loadmat(matfile)
    spiketrain = mat["spiketrain"].T
    # Separating coordinates and timestamps.
    spikes, timestamps = spiketrain[:, :-1], spiketrain[:, -1]
    # Getting events addresses.
    # First entry -> Event number.
    # Second entry -> Event address in [0,100).
    events_nums, events_addrs = spikes.nonzero()
    # Mapping addresses to 2D coordinates.
    # The mapping is (x%address, y//address), from the paper.
    events = np.zeros((len(events_nums)), dtype=dtype)
    events["x"] = events_addrs % sensor_size[0]
    events["y"] = events_addrs // sensor_size[1]
    # Converting floating point seconds to integer microseconds.
    events["t"] = (timestamps[events_nums] * 1e6).astype(int)
    # Converting -1 polarities to 0.
    events["p"] = np.max(spikes[(events_nums, events_addrs)], 0).astype(int)
    return events


#####
# Dataset
#####


def stmnist(
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> IterDataPipe:
    """
    Events have (xytp) ordering.
    Parameters:
        root (string): path to the ZIP archive downloaded by the user (e.g. "./STMNIST.zip").
        transform (callable, optional): a callable of transforms to be applied to events data.
        target_transform (callable, optional): a callable of transform to be applied to target data.
    Returns:
        dp (IterDataPipe): Torchdata data pipe that yields a tuple of events (or transformed events) and target.
    """
    # The root is specified as "directory/archive.zip".
    assert check_integrity(root, MD5), "The ZIP archive is not present or it is corrupted."
    # Creating the datapipe.
    dp = FileLister(root=root)
    dp = FileOpener(dp, mode="b")
    # Unzipping the archive.
    dp = ZipArchiveLoader(dp)
    # Filtering only MAT files (discarding the LUT).
    dp = Filter(dp, _is_mat_file)
    # Separating file path and file data.
    fpath_dp, fdata_dp = UnZipper(dp, sequence_length=2)
    # Reading data to structured NumPy array.
    event_dp = fdata_dp.map(_spiketrain_to_array)
    # Extracting label from file path.
    label_dp = fpath_dp.map(_read_label_from_filepath)
    if transform:
        event_dp = event_dp.map(transform)
    if target_transform:
        label_dp = label_dp.map(target_transform)
    # Zipping events and target in tuple.
    dp = Zipper(event_dp, label_dp)
    return dp
