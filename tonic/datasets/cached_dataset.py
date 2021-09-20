import os
from typing import Callable, Optional, Tuple, Iterable, Union
from warnings import warn
from dataclasses import dataclass
import h5py
from pathlib import Path
import hashlib


def save_to_cache(processed_data, fname: Union[str, Path]) -> None:
    """
    Save data to cache
    Args:
        processed_data:
        fname:
    """
    with h5py.File(fname, "w") as f:
        if type(processed_data) == tuple:  # can be events, frames, imu, gps, etc.
            for j, data_piece in enumerate(processed_data):
                f.create_dataset(f'{j}', data=data_piece)
        else:
            f.create_dataset(str(0), data=processed_data)


def load_from_cache(fname: Union[str, Path]) -> Tuple:
    """
    Load data from cache
    Args:
        fname:
    Returns:
        data
    """
    with h5py.File(fname, "r") as f:
        data = tuple(f[key][()] for key in f.keys())
    return data


@dataclass
class CachedDataset:
    dataset: Iterable
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    cache_path: str = "./cache/"

    def __post_init__(self):
        super().__init__()
        # Create cache directory
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)

    def __getitem__(self, item) -> (object, object):
        fname = self.cache_fname_from_index(item)
        try:
            data, target = load_from_cache(fname)
        except FileNotFoundError as _:
            warn(f"Data {item}: {fname} not in cache, generating it now", stacklevel=2)
            data, target = self.dataset[item]
            save_to_cache((data, target), fname=fname)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

    def cache_fname_from_index(self, item) -> Path:
        """
        Define a file naming scheme given an index of data
        Args:
            item:

        Returns:
            filename
        """
        transform_hash = hashlib.sha1(f"{self.transform}{self.target_transform}".encode()).hexdigest()
        return Path(self.cache_path) / f"{item}_{transform_hash}.h5"
