import tonic
from torchdata.datapipes.iter import (
    FileLister,
    Filter,
    Mapper,
    Forker,
    Zipper,
)
from functools import partial

def is_bin_file(data):
    return data.endswith('bin')

def read_label_from_filepath(filepath):
    return int(filepath.split('/')[-2])

def nmnist(root, train=False, transform=None, target_transform=None):
    dp = FileLister(root=root, recursive=True)
    dp = Filter(dp, is_bin_file)
    event_dp, label_dp = Forker(dp, num_instances=2)
    event_dp = Mapper(event_dp, partial(tonic.io.read_mnist_file, dtype=np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])))
    label_dp = Mapper(label_dp, read_label_from_filepath)
    if transform is not None:
        event_dp = Mapper(event_dp, transform)
    if target_transform is not None:
        label_dp = Mapper(label_dp, target_transform)
    dp = Zipper(event_dp, label_dp)
    return dp
