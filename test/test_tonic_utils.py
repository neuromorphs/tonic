import tonic
import tonic.transforms as transforms
from utils import create_random_input


class DummyDataset:
    def __init__(self, events, sensor_size, transform):
        self.events = events
        self.sensor_size = sensor_size
        self.transform = transform

    def __getitem__(self, index):
        return self.transform((self.events[index], self.sensor_size)), 1

    def __len__(self):
        return len(self.events)


def test_pytorch_batch_collation_dense_tensor():
    import torch

    events1, sensor_size = create_random_input()
    events2, sensor_size = create_random_input()

    events1, sensor_size = transforms.Downsample(time_factor=1e-3)(
        (events1, sensor_size)
    )
    events2, sensor_size = transforms.Downsample(time_factor=1e-3)(
        (events2, sensor_size)
    )

    transform = transforms.Compose([transforms.ToDenseTensor(merge_polarities=True)])
    dataset = DummyDataset(
        (events1[:5000], events2), sensor_size, transform
    )  # simulate recordings of different length
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=tonic.utils.pad_tensors, batch_size=batch_size
    )

    batch, label = next(iter(dataloader))

    max_time = int(events2["t"][-1]) + 1
    assert batch.shape[0] == max_time
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == 1


def test_pytorch_batch_collation_sparse_tensor():
    import torch

    events1, sensor_size = create_random_input()
    events2, sensor_size = create_random_input()

    transform = transforms.Compose([transforms.ToSparseTensor(merge_polarities=True)])
    dataset = DummyDataset(
        (events1[:5000], events2), sensor_size, transform
    )  # simulate recordings of different length
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=tonic.utils.pad_tensors, batch_size=batch_size
    )

    batch, label = next(iter(dataloader))

    max_time = int(events2["t"][-1]) + 1
    assert batch.shape[0] == max_time
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == 1

def test_plotting():
    events, sensor_size = create_random_input()
    
    tonic.utils.plot_event_grid(events)
