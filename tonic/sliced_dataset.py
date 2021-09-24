from dataclasses import dataclass
from typing import Iterable, List, Any

from .slicers import Slicer


@dataclass
class SlicedDataset:
    dataset: Iterable
    slicer: Slicer

    def __post_init__(self):
        # Generate slicing metadata
        self.metadata: List[Any] = []
        self.slice_dataset_map = []
        for dataset_index, (data, label) in enumerate(self.dataset):
            metadata = self.slicer.get_slice_metadata(data)
            self.metadata.append(metadata)
            self.slice_dataset_map += [(dataset_index, slice_index) for slice_index in range(len(metadata))]

    def __getitem__(self, item) -> Any:
        dataset_index, slice_index = self.slice_dataset_map[item]
        data, label = self.dataset[dataset_index]
        data_slice = self.slicer.slice_with_metadata(data, self.metadata[dataset_index])[slice_index]
        return data_slice, label

    def __len__(self):
        return len(self.slice_dataset_map)
