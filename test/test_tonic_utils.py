import tonic 
from utils import create_random_input


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

    ordering="xytp"
    (
        events1,
        orig_images,
        sensor_size,
        is_multi_image,
    ) = create_random_input(dtype)
    (
        events2,
        orig_images,
        sensor_size,
        is_multi_image,
    ) = create_random_input(dtype)

    transform = tonic.transforms.Compose([tonic.transforms.ToDenseTensor(ordering=ordering, sensor_size=sensor_size, merge_polarities=True)])
    dataset = DummyDataset((events1[:5000], events2), transform) # simulate recordings of different length
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=tonic.utils.pad_tensors, batch_size=batch_size)
    
    batch, label = next(iter(dataloader))
    
    max_time = int(events2[:,2][-1]) + 1
    assert batch.shape[0] == max_time
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == 1

def test_pytorch_batch_collation_sparse_tensor():
    import torch

    ordering="xytp"
    (
        events1,
        orig_images,
        sensor_size,
        is_multi_image,
    ) = create_random_input(dtype)
    (
        events2,
        orig_images,
        sensor_size,
        is_multi_image,
    ) = create_random_input(dtype)

    transform = tonic.transforms.Compose([tonic.transforms.ToSparseTensor(ordering=ordering, sensor_size=sensor_size, merge_polarities=True)])
    dataset = DummyDataset((events1[:5000], events2), transform) # simulate recordings of different length
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=tonic.utils.pad_tensors, batch_size=batch_size)
    
    batch, label = next(iter(dataloader))
    
    max_time = int(events2[:,2][-1]) + 1
    assert batch.shape[0] == max_time
    assert batch.shape[1] == batch_size
    assert batch.shape[2] == 1
