from tonic.datasets import POKERDVS
from tonic import SlicedDataset
from tonic.slicers import SliceByEventCount


def test_sliced_dataset():
    dataset = POKERDVS(save_to="./data/", train=False)

    target_number = 0
    for data, label in dataset:
        target_number += len(data) // 200

    slicer = SliceByEventCount(event_count=200)
    sliced_dataset = SlicedDataset(dataset, slicer, metadata_path="./cache/metadata")

    for data, label in sliced_dataset:
        assert len(data) == 200

    assert len(sliced_dataset) == target_number
