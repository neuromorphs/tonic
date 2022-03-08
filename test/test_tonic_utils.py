import tonic
import tonic.transforms as transforms
from utils import create_random_input
from sys import platform


class DummyDataset:
    def __init__(self, events, transform):
        self.events = events
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.events[index]), 1

    def __len__(self):
        return len(self.events)


def test_pytorch_batch_collation_dense_tensor():
    import torch

    events1, sensor_size = create_random_input()
    events2, sensor_size = create_random_input()

    time_window = 1000
    transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)]
    )
    dataset = DummyDataset(
        (events1[:5000], events2), transform
    )  # simulate recordings of different length
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=tonic.collation.PadTensors(), batch_size=batch_size
    )

    batch, label = next(iter(dataloader))

    max_time = int(events2["t"][-1])
    assert batch.shape[0] == max_time // time_window
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == sensor_size[2]


def test_plotting():
    events, sensor_size = create_random_input()
    # this test doesn't finish on Windows
    if platform == "linux":
        tonic.utils.plot_event_grid(events)
