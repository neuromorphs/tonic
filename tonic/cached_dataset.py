import os
import shutil
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
        data: numpy ndarray-like or a list thereof.
        targets: same as data, can be None.
        file_path: caching file path.
    """
    with h5py.File(file_path, "w") as f:
        for name, data in zip(["data", "target"], [data, targets]):
            if type(data) != tuple:
                data = (data,)
            # can be events, frames, imu, gps, target etc.
            for i, data_piece in enumerate(data):
                if type(data_piece) == dict:
                    for key, item in data_piece.items():
                        f.create_dataset(
                            f"{name}/{i}/{key}",
                            data=item,
                            compression="lzf" if type(item) == np.ndarray else None,
                        )
                else:
                    f.create_dataset(
                        f"{name}/{i}",
                        data=data_piece,
                        compression="lzf" if type(data_piece) == np.ndarray else None,
                    )


def load_from_cache(file_path: Union[str, Path]) -> Tuple:
    """
    Load data from file cache, separately for (data) and (targets). Can assemble dictionaries back together.
    Args:
        file_path: caching file path.
    Returns:
        data, targets
    """
    data_list = []
    target_list = []
    with h5py.File(file_path, "r") as f:
        for name, _list in zip(["data", "target"], [data_list, target_list]):
            for index in f[name].keys():
                if hasattr(f[f"{name}/{index}"], "keys"):
                    data = {
                        key: f[f"{name}/{index}/{key}"][()]
                        for key in f[f"{name}/{index}"].keys()
                    }
                else:
                    data = f[f"{name}/{index}"][()]
                _list.append(data)
    if len(data_list) == 1:
        data_list = data_list[0]
    if len(target_list) == 1:
        target_list = target_list[0]
    return data_list, target_list


@dataclass
class CachedDataset:
    """
    CachedDataset caches the data samples to the hard drive for subsequent reads, thereby potentially improving data loading speeds.
    If dataset is None, then the length of this dataset will be inferred from the number of files in the caching folder. Pay
    attention to the cache path you're providing, as CachedDataset will simply check if there is a file present with the index that
    it is looking for. When using train/test splits, it is wise to also take that into account in the cache path.

    Parameters:
        dataset:
            Dataset to be cached. Can be None, if only files in cache_path should be used.
        cache_path:
            The preferred path where the cache will be written to and read from.
        reset_cache:
            When True, will clear out the cache path during initialisation. Default is False
        transform:
            Transforms to be applied on the data
        target_transform:
            Transforms to be applied on the label/targets
        num_copies:
            Number of copies of each sample to be cached.
            This is a useful parameter if the dataset is being augmented with slow, random transforms.
    """

    dataset: Iterable
    cache_path: str
    reset_cache: bool = False
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    num_copies: int = 1

    def __post_init__(self):
        super().__init__()
        # Create cache directory
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)
        if self.reset_cache:
            shutil.rmtree(self.cache_path)
            os.makedirs(self.cache_path)
        if self.dataset is None:
            self.n_samples = (
                len(
                    [
                        name
                        for name in os.listdir(self.cache_path)
                        if os.path.isfile(os.path.join(self.cache_path, name))
                        and name.endswith(".hdf5")
                    ]
                )
                // self.num_copies
            )
        else:
            self.n_samples = len(self.dataset)

    def __getitem__(self, item) -> (object, object):
        copy = np.random.randint(self.num_copies)
        file_path = os.path.join(self.cache_path, f"{item}_{copy}.hdf5")
        try:
            data, targets = load_from_cache(file_path)
        except (FileNotFoundError, OSError) as _:
            logging.info(
                f"Data {item}: {file_path} not in cache, generating it now",
                stacklevel=2,
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
