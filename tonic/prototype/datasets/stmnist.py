from torchdata.datapipes.iter import (
    FileLister,
    Filter,
    Forker,
    Zipper,
)
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from tonic.download_utils import extract_archive, check_integrity

#####
# Dataset properties.
#####

sensor_size = (10, 10, 2)
dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
MD5 = "2eef16be7356bc1a8f540bb3698c4e0ei" 

#####
# Functions
#####


def _is_mat_file(filename):
    return filename.endswith("mat") and "LUT" not in filename


def _read_label_from_filepath(filepath):
    return int(filepath.split("/")[-2])


def _at_least_n_files(root, n_files, file_type):
    check = n_files <= len(list(Path(root).glob(f"**/*{file_type}")))
    return check


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


def _check_exists(filepath, md5):
    check = check_integrity(filepath, md5) 
    check &= _at_least_n_files(filepath, n_files=100, file_type=".mat")
    return check


#####
# Dataset
#####


def stmnist(root, transform=None, target_transform=None):
    # The root is specified as "directory/archive.zip". 
    # We strip 'archive.zip' and get only 'directory', where it is
    # extracted in 'data_submission'. 
    filepath = root[::-1].split("/", 1)[-1][::-1] + "/data_submission"
    # Extracting the ZIP archive.
    if not _check_exists(filepath, MD5):
        extract_archive(root)
    # Creating the datapipe.
    dp = FileLister(root=filepath, recursive=True)
    dp = Filter(dp, _is_mat_file)
    event_dp, label_dp = Forker(dp, num_instances=2)
    event_dp = event_dp.map(_spiketrain_to_array)
    label_dp = label_dp.map(_read_label_from_filepath)
    if transform:
        event_dp = event_dp.map(transform)
    if target_transform:
        label_dp = label_dp.map(target_transform)
    dp = Zipper(event_dp, label_dp)
    return dp
