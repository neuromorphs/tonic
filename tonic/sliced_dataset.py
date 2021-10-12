from dataclasses import dataclass
from typing import Iterable, List, Any, Optional, Callable

from .slicers import Slicer


@dataclass
class SlicedDataset:
    """The primary use case for a SlicedDataset is to cut existing examples in a dataset 
    into smaller chunks. For that it takes a regular dataset and a slicing method as input. 
    It then generates metadata about the slices and where to find them in each original sample.
    The new dataset length will be the sum of all slices for each sample.
    
    Parameters:
        dataset: a dataset object which implements __getitem__ and __len__ methods.
        slicer: a function which implements the tonic.slicers.Slicer protocol, meaning that 
                it doesn't have to inherit from it but implement all its methods.
    """

    dataset: Iterable
    slicer: Slicer
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None

    def __post_init__(self):
        # Generate slicing metadata
        self.metadata: List[Any] = []
        self.slice_dataset_map = []
        for dataset_index, (data, targets) in enumerate(self.dataset):
            metadata = self.slicer.get_slice_metadata(data)
            self.metadata.append(metadata)
            self.slice_dataset_map += [
                (dataset_index, slice_index) for slice_index in range(len(metadata))
            ]

    def __getitem__(self, item) -> Any:
        dataset_index, slice_index = self.slice_dataset_map[item]
        data, targets = self.dataset[dataset_index]
        data_slice = self.slicer.slice_with_metadata(
            data, self.metadata[dataset_index]
        )[slice_index]
        if self.transform:
            data_slice = self.transform(data_slice)
        if self.target_transform:
            targets = self.target_transform(targets)
        return data_slice, targets

    def __len__(self):
        return len(self.slice_dataset_map)
