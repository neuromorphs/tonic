from tonic import datasets
import tonic.transforms as transforms
from tonic import CachedDataset


def test_caching_pokerdvs():
    dataset = datasets.POKERDVS(
        save_to="./data",
        train=False,
        download=True
    )
    dataset = CachedDataset(dataset)
    for data, label in dataset:
        print(data.shape, label)


def test_caching_transforms():
    sensor_size = datasets.POKERDVS.sensor_size
    preprocess = transforms.Compose([transforms.Downsample(time_factor=1, spatial_factor=1, ordering="xytp", sensor_size=sensor_size)])
    augmentation = transforms.Compose([transforms.Downsample(time_factor=1, spatial_factor=1, ordering="xytp", sensor_size=sensor_size)])
    dataset = datasets.POKERDVS(
        save_to="./data",
        train=False,
        download=True,
        transform=preprocess,
    )

    dataset_cached = CachedDataset(dataset, transform=augmentation, num_copies=4)

    print(dataset_cached)
    for (data, label), (data2, label2) in zip(dataset, dataset_cached):
        assert (data == data2).all()
        assert label == label2
