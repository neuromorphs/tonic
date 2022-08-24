import os
import shutil

from tonic import SlicedDataset
from tonic.datasets import POKERDVS
from tonic.slicers import SliceByEventCount


def test_sliced_dataset():
    dataset = POKERDVS(save_to="./data/", train=False)

    metadata_path = "./cache/metadata"
    if os.path.exists(metadata_path):
        shutil.rmtree(metadata_path)

    target_number = 0
    for data, label in dataset:
        target_number += len(data) // 200

    slicer = SliceByEventCount(event_count=200)
    sliced_dataset = SlicedDataset(dataset, slicer, metadata_path=metadata_path)

    for data, label in sliced_dataset:
        assert len(data) == 200
        assert type(label) == int

    assert len(sliced_dataset) == target_number
