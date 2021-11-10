import os
import shutil
import h5py
import numpy as np
from pathlib import Path
from tonic import datasets
import tonic.transforms as transforms
from tonic import CachedDataset


def test_caching_pokerdvs():
    dataset = datasets.POKERDVS(save_to="./data", train=False)
    dataset = CachedDataset(dataset, cache_path="./cache/cache1")
    for data, label in dataset:
        print(data.shape, label)

def test_caching_transforms():
    sensor_size = datasets.POKERDVS.sensor_size
    preprocess = transforms.Compose(
        [transforms.Downsample(time_factor=1, spatial_factor=1)]
    )
    augmentation = transforms.Compose(
        [transforms.Downsample(time_factor=1, spatial_factor=1)]
    )
    dataset = datasets.POKERDVS(save_to="./data", train=True, transform=preprocess)

    dataset_cached = CachedDataset(dataset, cache_path="./cache/cache2", transform=augmentation, num_copies=4)

    for (data, label), (data2, label2) in zip(dataset, dataset_cached):
        assert (data == data2).all()
        assert label == label2

def test_caching_reset():
    cache_path = Path('./cache/test3')
    if cache_path.exists():
        shutil.rmtree(cache_path)

    # simulate outdated cache file
    old_file = cache_path / '0_0.hdf5'
    dummy_content = np.zeros((3,3))
    os.makedirs(cache_path)
    with h5py.File(old_file, "w") as f:
        f.create_dataset(f"data/0", data=dummy_content)
        f.create_dataset(f"target/0", data=dummy_content)
    
    dataset = datasets.POKERDVS(save_to="./data", train=False)

    # load first sample from cache, which will load the old file
    dataset_cached = CachedDataset(dataset, cache_path=cache_path, reset_cache=False)
    data, target = dataset_cached[0]
    assert (data == dummy_content).all()
    
    # when using reset_cache=True, the folder will be reset and the right file cached
    dataset_cached = CachedDataset(dataset, cache_path=cache_path, reset_cache=True)
    data, target = dataset_cached[0]
    assert data.shape != dummy_content.shape
    assert data.shape[0] > 100
