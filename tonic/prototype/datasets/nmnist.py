from torchdata.datapipes.iter import (
    FileLister,
    Filter,
    Mapper,
    Forker,
    Zipper,
)
from functools import partial
import tonic
import os
import numpy as np
from pathlib import Path
from tonic.io import read_mnist_file
from tonic.download_utils import check_integrity, download_and_extract_archive

#####
# Dataset properties.
#####

SENSOR_SIZE = (32, 32, 2)
DTYPE = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
BASE_URL = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
TRAIN_URL = BASE_URL + "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded"
TRAIN_FILENAME = "train.zip"
TRAIN_MD5 = "20959b8e626244a1b502305a9e6e2031"
TRAIN_FOLDER = "train"
TEST_URL = BASE_URL + "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded"
TEST_FILENAME = "test.zip"
TEST_MD5 = "69ca8762b2fe404d9b9bad1103e97832"
TEST_FOLDER = "test"

#####
# Functions
#####


def is_bin_file(data):
    return data.endswith("bin")


def read_label_from_filepath(filepath):
    return int(filepath.split("/")[-2])


def at_least_n_files(root, n_files, file_type):
    check = n_files <= len(list(Path(root).glob(f"**/*{file_type}")))
    return check


def check_exists(filepath):
    check = False
    check = check_integrity(filepath)
    check = at_least_n_files(filepath, n_files=1000, file_type=".bin")
    return check


def first_saccade_filter(events):
    return events[events["t"] < 1e5]


#####
# Dataset
#####


def nmnist(
    root, transform=None, target_transform=None, train=True, first_saccade_only=False
):
    # Setting file path depending on train value.
    filepath = root + "/" + (TRAIN_FOLDER if train else TEST_FOLDER)
    url = TRAIN_URL if train else TEST_URL
    md5 = TRAIN_MD5 if train else TEST_MD5
    # Downloading the MNIST file if it exists.
    if not check_exists(filepath):
        download_and_extract_archive(url=url, download_root=filepath, md5=md5)
    # Creating the datapipe.
    dp = FileLister(root=filepath, recursive=True)
    dp = Filter(dp, is_bin_file)
    # Thinking about avoiding this fork in order to apply transform to both targe and events.
    event_dp, label_dp = Forker(dp, num_instances=2)
    event_dp = Mapper(event_dp, partial(read_mnist_file, dtype=DTYPE))
    if first_saccade_only:
        event_dp = Mapper(event_dp, first_saccade_filter)
    label_dp = Mapper(label_dp, read_label_from_filepath)
    if transform is not None:
        event_dp = Mapper(event_dp, transform)
    if target_transform is not None:
        label_dp = Mapper(label_dp, target_transform)
    dp = Zipper(event_dp, label_dp)
    return dp