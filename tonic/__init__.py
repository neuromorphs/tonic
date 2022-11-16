from pbr.version import VersionInfo

from . import collation, datasets, io, prototype, slicers, transforms, utils
from .cached_dataset import CachedDataset, DiskCachedDataset, MemoryCachedDataset
from .dataset import Dataset
from .sliced_dataset import SlicedDataset

all = "__version__"
__version__ = VersionInfo("tonic").release_string()
