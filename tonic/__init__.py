from pbr.version import VersionInfo

from . import collation, datasets, io, slicers, transforms, utils
from .cached_dataset import CachedDataset, DiskCachedDataset, MemoryCachedDataset
from .dataset import Dataset
from .sliced_dataset import SlicedDataset

try:
    from . import prototype
except (ImportError, NameError):
    pass

all = "__version__"
__version__ = VersionInfo("tonic").release_string()
