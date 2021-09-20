import pytest
from tonic import datasets
import tonic.transforms as transforms
from tonic.datasets.cached_dataset import CachedDataset


def test_caching_nmnist():
    dataset = datasets.POKERDVS(
        save_to="./data",
        train=False,
        download=True
    )
    dataset = CachedDataset(dataset)
    for data, label in dataset:
        print(data.shape, label)


@pytest.mark.skip(reason="transforms are not yet compatible.")
def test_caching_transforms():
    preprocess = transforms.Compose([transforms.Downsample(time_factor=1e-3)])
    augmentation = transforms.Compose([transforms.Downsample(time_factor=1e-3)])
    dataset = datasets.POKERDVS(
        save_to="./data",
        train=False,
        download=True,
        transform=preprocess,
    )
    dataset_cached = CachedDataset(dataset, transform=augmentation)

    print(dataset)
    for (data, label), (data2, label2) in zip(dataset, dataset_cached):
        print(data[:, 0], label)
        print(data2[:, 0], label2)
