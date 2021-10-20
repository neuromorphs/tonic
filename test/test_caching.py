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
