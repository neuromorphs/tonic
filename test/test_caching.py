import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from tonic import DiskCachedDataset, MemoryCachedDataset, datasets, transforms


def test_memory_caching_pokerdvs():
    dataset = datasets.POKERDVS(save_to="./data", train=False)
    cached_dataset = MemoryCachedDataset(dataset)
    assert len(dataset) == len(cached_dataset)

    # cache all samples
    for data, label in cached_dataset:
        pass

    assert len(cached_dataset.samples_dict) == len(dataset)


def test_caching_pokerdvs():
    dataset = datasets.POKERDVS(save_to="./data", train=False)
    cache_path = "./cache/test1"
    cached_dataset = DiskCachedDataset(dataset, cache_path)

    # cache all samples
    for data, label in cached_dataset:
        pass

    assert len(dataset) == len(cached_dataset)
    assert len(os.listdir(cache_path)) == len(dataset)


def test_caching_transforms():
    sensor_size = datasets.POKERDVS.sensor_size
    preprocess = transforms.Compose(
        [transforms.Downsample(time_factor=1, spatial_factor=1)]
    )
    augmentation = transforms.Compose(
        [transforms.Downsample(time_factor=1, spatial_factor=1)]
    )
    dataset = datasets.POKERDVS(save_to="./data", train=True, transform=preprocess)

    dataset_cached = DiskCachedDataset(
        dataset, cache_path="./cache/test2", transform=augmentation, num_copies=4
    )

    for (data, label), (data2, label2) in zip(dataset, dataset_cached):
        assert (data == data2).all()
        assert label == label2


def test_caching_reset():
    cache_path = Path("./cache/test3")
    if cache_path.exists():
        shutil.rmtree(cache_path)

    # simulate outdated cache file
    old_file = cache_path / "0_0.hdf5"
    dummy_content = np.zeros((3, 3))
    os.makedirs(cache_path)
    with h5py.File(old_file, "w") as f:
        f.create_dataset(f"data/0", data=dummy_content)
        f.create_dataset(f"target/0", data=dummy_content)

    dataset = datasets.POKERDVS(save_to="./data", train=False)

    # load first sample from cache, which will load the old file
    dataset_cached = DiskCachedDataset(
        dataset, cache_path=cache_path, reset_cache=False
    )
    data, target = dataset_cached[0]
    assert (data == dummy_content).all()

    # when using reset_cache=True, the folder will be reset and the right file cached
    dataset_cached = DiskCachedDataset(dataset, cache_path=cache_path, reset_cache=True)
    data, target = dataset_cached[0]
    assert data.shape != dummy_content.shape
    assert data.shape[0] > 100


def test_caching_from_files():
    dataset = datasets.POKERDVS(save_to="./data", train=False)
    cache_path = "./cache/test4"
    dataset = DiskCachedDataset(dataset, cache_path)

    n_cached_samples = 10
    for i in range(n_cached_samples):
        events, target = dataset[i]

    dataset = DiskCachedDataset(dataset=None, cache_path=cache_path)
    assert len(dataset) == n_cached_samples
    assert len(os.listdir(cache_path)) == len(dataset)

    # Make sure iteration stops properly when available number of items is reached.
    for item in dataset:
        events, target = item
