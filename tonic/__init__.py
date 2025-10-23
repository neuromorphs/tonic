try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

from . import collation, datasets, io, slicers, transforms, utils
from .cached_dataset import (
    Aug_DiskCachedDataset,
    CachedDataset,
    DiskCachedDataset,
    MemoryCachedDataset,
)
from .dataset import Dataset
from .sliced_dataset import SlicedDataset

try:
    __version__ = version("tonic")
except PackageNotFoundError:
    # Package not installed, use fallback
    __version__ = "unknown"

__all__ = [
    "__version__",
    "collation",
    "datasets",
    "io",
    "slicers",
    "transforms",
    "utils",
    "Aug_DiskCachedDataset",
    "CachedDataset",
    "DiskCachedDataset",
    "MemoryCachedDataset",
    "Dataset",
    "SlicedDataset",
]
