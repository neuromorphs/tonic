Release notes
=============

1.0.0
-----
This is a major release where we focused on performance improvements. In the interest of future maintainability, there are a few breaking changes.

* dataset downloads and instantiation: The `download` parameter has been removed from all datasets. When previously we did slow hashing of files to verify strict file integrity, this is now only done once after the download is completed. The next time the same dataset is instantiated, only lightweight checks are going to be run to check if the right filenames and number of files are present. This considerably speeds up training scripts overall.
* renamed some stochastic transforms for better consistency
* default download path for datasets has changed. For previous Tonic users, it is recommended to just clear your data directory and download datasets again to avoid duplication on your hard drive.
* support for slicing and caching. Using CachedDataset or SlicedDataset wrapper classes, recordings can now be chunked into smaller segments and/or written to disk in an efficient format for fast loading. Use of dtypes under the hood to keep disk footprint to a minimum.
* removed ToSparseTensor transform to avoid confusion with ToFrame and because PyTorch/Tensorflow support is not there yet. Will reintroduce once support for sparse computation is there in major frameworks.
* new `io` module which contains a list of parsers for different neuromorphic file formats.
* batch collation functionality is now to be found under `tonic.collation`
* full compatibility with PyTorch vision/audio transforms. Tonic transforms now only take in and output one object.
* code coverage, documentation improvements, bug fixes

0.4.6
-----
* sensor_size is now being passed from transform to transform, in case it is adjusted. Important for cropping and spatial resizing.
* test suite improvements

0.4.5
-----
* Dropped Subsample transform in favour of Downsample transform, which can now scale timestamps and spatial coordinates in one transform together. 
* Renamed `DropEvents` transform to `DropEvent` to be consistent with `DropPixel` and other singular transform names.
* NCALTECH101 events are now returned as floats. 
* MaskHotPixel has been incorporated into DropPixel transform, which can now either automatically suppress hot pixels or a fixed list of coordinates. 

0.4.0
-----
Tonic is now free from direct PyTorch and PyTorch Vision dependencies, meaning that if someone wanted to use it with another pipeline (e.g. TensorFlow), they would be able to do so. This release is the first step to making Tonic more flexible in that respect. For previous users that mainly means that ``tonic.datasets.DataLoader`` is no longer available anymore, which had been an alias for the pytorch dataloader. Now you will have to import the class directly like so:
::

  import torch
  dataloader = torch.utils.data.DataLoader(dataset)
