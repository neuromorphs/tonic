import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import h5py

from .slicers import Slicer


def save_metadata(path, metadata):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "slice_metadata.h5")
    with h5py.File(file_path, "w") as f:
        for i, data in enumerate(metadata):
            f.create_dataset(f"metadata_{i}", data=data)
    print(f"Metadata written to {file_path}.")


def load_metadata(path):
    file_path = os.path.join(path, "slice_metadata.h5")
    with h5py.File(file_path, "r") as f:
        metadata = [f[f"metadata_{i}"][()] for i in range(len(f.keys()))]
    print(f"Metadata read from {file_path}.")
    return metadata


@dataclass
class SlicedDataset:
    """The primary use case for a SlicedDataset is to cut existing examples in a dataset into
    smaller chunks. For that it takes an iterable dataset and a slicing method as input. It then
    generates metadata about the slices and where to find them in each original sample. The new
    dataset length will be the sum of all slices across samples.

    Parameters:
        dataset: a dataset object which implements __getitem__ and __len__ methods.
        slicer: a function which implements the tonic.slicers.Slicer protocol, meaning that
                it doesn't have to inherit from it but implement all its methods.
        metadata_path: filepath where slice metadata should be stored, so that it does not
                       have to be recomputed the next time. If None, will be recomputed
                       every time.
        transform: Transforms to be applied on the data
        target_transform: Transforms to be applied on the label/targets
        transforms: A callable of transforms that is applied to both data and labels at the same time.
    """

    dataset: Iterable
    slicer: Slicer
    metadata_path: Optional[str] = None
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    transforms: Optional[Callable] = None

    def __post_init__(self):
        """Will try to read metadata from disk to know where slices start and stop for each sample.

        If no metadata_path is provided or no file slice_metadata.h5 is found in that path,
        metadata will be generated from scratch.
        """
        if self.metadata_path:
            try:
                self.metadata = load_metadata(self.metadata_path)
            except (FileNotFoundError, OSError) as _:
                self.metadata = self.generate_metadata()
                save_metadata(self.metadata_path, self.metadata)
        else:
            self.metadata = self.generate_metadata()
        self.slice_dataset_map = [
            (sample_index, slice_index)
            for sample_index, sample_metadata in enumerate(self.metadata)
            for slice_index in range(len(sample_metadata))
        ]

    def generate_metadata(self):
        """Slices every sample in the wrapped dataset and returns start and stop metadata for each
        slice."""
        return [
            self.slicer.get_slice_metadata(data, targets)
            for data, targets in self.dataset
        ]

    def __getitem__(self, item) -> Any:
        dataset_index, slice_index = self.slice_dataset_map[item]
        data, targets = self.dataset[dataset_index]
        data_slice, target_slice = self.slicer.slice_with_metadata(
            data, targets, [self.metadata[dataset_index][slice_index]]
        )
        if isinstance(data_slice, list) and len(data_slice) == 1:
            data_slice = data_slice[0]
        if isinstance(target_slice, list) and len(target_slice) == 1:
            target_slice = target_slice[0]
        if self.transform:
            data_slice = self.transform(data_slice)
        if self.target_transform:
            target_slice = self.target_transform(target_slice)
        if self.transforms is not None:
            data_slice, target_slice = self.transforms(data_slice, target_slice)
        return data_slice, target_slice

    def __len__(self):
        return len(self.slice_dataset_map)
