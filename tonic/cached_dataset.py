import os
from typing import Callable, Optional, Tuple, Iterable, Union
from warnings import warn
from dataclasses import dataclass
import h5py
from pathlib import Path
import hashlib
import numpy as np
import logging


def save_to_cache(data, targets, file_path: Union[str, Path]) -> None:
    """
    Save data to caching path on disk in an hdf5 file. Can deal with data
    that is a dictionary.
    Args:
        data: numpy ndarray-like, a list or dictionary thereof.
        targets: same as data, can be None.
        file_path: caching file path.
    """
    with h5py.File(file_path, "w") as f:
        # can be events, frames, imu, gps, target etc.
        if type(data) != tuple:
            data = (data,)
        for i, data_piece in enumerate(data):
            if type(data_piece) == dict:
                for key, item in data_piece.items():
                    f.create_dataset(f"data/{i}/{key}", data=item, compression='lzf')
            else:
                f.create_dataset(f"data/{i}", data=data_piece, compression='lzf')
        if type(targets) != tuple:
            targets = (targets,)
        for j, target_piece in enumerate(targets):
            if type(target_piece) == dict:
                for key, item in target_piece.items():
                    f.create_dataset(f"target/{j}/{key}", data=item)
            else:
                f.create_dataset(f"target/{j}", data=target_piece)


def load_from_cache(file_path: Union[str, Path]) -> Tuple:
    """
    Load data from cache
    Args:
        file_path:
    Returns:
        data
    """
    data_list = []
    target_list = []
    with h5py.File(file_path, "r") as f:
        for index in f['data'].keys():
            if hasattr(f[f"data/{index}"], "keys"):
                data = {key: f[f"data/{index}/{key}"][()] for key in f[f"data/{index}"].keys()}
            else:
                data = f[f"data/{index}"][()]
            data_list.append(data)
        for index in f['target'].keys():
            if hasattr(f[f"target/{index}"], "keys"):
                target = {key: f[f"target/{index}/{key}"][()] for key in f[f"target/{index}"].keys()}
            else:
                target = f[f"target/{index}"][()]
            target_list.append(target)
    return data_list, target_list


@dataclass
class CachedDataset:
    """
    CachedDataset caches the data samples to the hard drive for subsequent reads, thereby potentially improving data loading speeds.
    If dataset is None, then the length of this dataset will be inferred from the number of files in the caching folder. 

    Parameters:
        dataset:
            Dataset to be cached. Can be None, if only files in cache_path should be used.
        cache_path:
            The preferred path where the cache will be written to and read from. Default is ./cache
        transform:
            Transforms to be applied on the data
        target_transform:
            Transforms to be applied on the label/targets
        num_copies:
            Number of copies of each sample to be cached.
            This is a useful parameter if the dataset is being augmented with slow, random transforms.
    """

    dataset: Optional[Iterable] = None
    cache_path: str
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    num_copies: int = 1

    def __post_init__(self):
        super().__init__()
        # Create cache directory
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)
        if self.dataset is None:
            self.n_samples = len([name for name in os.listdir(self.cache_path) if os.path.isfile(name) and name.endswith('.hdf5')]) // self.num_copies
        else:
            self.n_samples = len(self.dataset)
            

    def __getitem__(self, item) -> (object, object):
        copy = np.random.randint(self.num_copies)
        file_path = os.path.join(self.cache_path, f"{item}_{copy}.hdf5")
        try:
            data, targets = load_from_cache(file_path)
        except FileNotFoundError as _:
            logging.info(
                f"Data {item}: {file_path} not in cache, generating it now", stacklevel=2
            )
            data, targets = self.dataset[item]
            save_to_cache(data, targets, file_path=file_path)
            # format might change during save to hdf5,
            # i.e. tensors -> np arrays
            data, targets = load_from_cache(file_path)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            targets = self.target_transform(targets)
        return data, targets

    def __len__(self):
        return self.n_samples
