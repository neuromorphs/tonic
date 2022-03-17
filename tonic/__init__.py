from . import datasets, transforms, collation, io, slicers, utils
from .dataset import Dataset
from .cached_dataset import CachedDataset, DiskCachedDataset, MemoryCachedDataset
from .sliced_dataset import SlicedDataset
from pbr.version import VersionInfo

all = ('__version__')
__version__ = VersionInfo('tonic').release_string()