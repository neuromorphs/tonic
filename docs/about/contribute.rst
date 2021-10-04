Contribute
==========

How do I add a new transformation?
----------------------------------
To add a new transform, please consider all the 3 following steps, not necessarily in this order:

* Add a new functional transform using one of the supported backends such as numpy or torch in `tonic/functional`
* Add a transform wrapper class in `tonic/transforms`.
* Add a test for the functional transform and sufficient assertions to check its validity.

Current transforms work on events and/or images, but transforms on other data such as IMU are possible as well. Identify what data type your transform targets and pass it to your function. You will most likely need `sensor_size` and `ordering` parameters as well. If you just want to add another parameter to an existing transform, consider if what you want to do might be applied more generally and therefore justify its own transform.

When implementing the transform, try to make use of existing transforms if possible. That helps to reduce duplicate code. Please spell out variable names to `make it easier <https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343>`_ for people to understand your code. For example, `y_index` is preferable to `yi` or `compressed_images` to `comp_imgs`. Add a docstring to your transform and explain the function and its parameters.

The transform wrapper class serves as an abstraction layer. In a later stage, maybe this function will automatically choose between a torch or numpy implementation for example. For the most part you don't have to worry about it too much for now, but please add the class.

One of Tonic's goals is to be a reliable package that doesn't break. As such it is important to have a strong test suite in place. Please consider making use of the parameterized injections to test your transform covering all possible parameters with at least one injection, better even from different ranges and combinations thereof.


How can I add a new dataset?
----------------------------
If you want to add a new dataset, there are 2 main questions to consider:

* Where is the dataset hosted?
* What format is it in?

To download a dataset, Tonic makes use of a copy of pytorch vision's tools that can be found `here <https://github.com/neuromorphs/tonic/blob/develop/tonic/datasets/download_utils.py>`_.
Cloud providers such as Dropbox, Google Drive, OneDrive etc. are usually not straightforward to automatically download from, so if the dataset owner provides multiple mirrors,
prefer a simple REST-like webpage such as https://www.neuromorphic-vision.com/public/downloads/.


The trickier question is the dataset decoding. There exist a variety of file formats that events and related data are saved in, including .txt, .aedat, .dat, .es, .mat, .hdf5, .rosbag and more.
Ordered by degree of easiness to decode a file format, here is one subjective ranking:

1. **hdf5**. This bundles everything in a single file which prevents it from getting messy on your file system and the h5py python package is reliable and a breeze to use. Easy!
2. **rosbag**. This is a popular format used by roboticists which similarly bundles everything in a single file. The downside however that there is lacking support for a stand-alone python package to decode these files. Tonic builds on the `importRosbag <https://github.com/event-driven-robotics/importRosbag>`_ package which does a great job at decoding, unless your .rosbags are compressed (as of v1.0.3). Plus it might not be able to decode all the topics in there. Good with potential hickups!
3. **npy** and **mat**. These are file formats that can be handled with ubiquitious numpy and scipy packges. Although they might not be the most compact format, they are still a good solution due to easiness of decoding.
4. **aedat**, **dat** and **es**. Every lab has their own file format because why not! But support to decode those files in a reliable python package is often scarce. The `Loris <https://github.com/neuromorphic-paris/loris>`_ and `aedat <https://github.com/neuromorphicsystems/aedat>`_ packages are trying their best, but might not be maintained at all times... Plus when a dataset is published, there is no reason why it has to be split in single files. Avoid if possible
5. **txt**. Ever tried decoding 1 million events by splitting strings and converting them to floats? No? I did it and I can tell you that it's incredibly slow... Avoid at all costs.

The way a dataset class is organised generally follows this structure:

- The ``__init__`` function takes the necessary parameters. If the dataset has a train/test split, you can have a `train` parameter. Or it might be split into scenes, or maybe no split at all.
  If the dataset consists of many individual files, you can save the path to each file, so that you don't have to read the whole dataset into memory at once.
- The ``download`` function makes use of the torchvision tools to download and extract archives / files to the right location. Be sure to have md5 hashes ready for the files so that Tonic can check them for consistency.
- The ``__len__`` function provides the number of samples in your dataset.
- ``__getitem__`` is where the interesting stuff happens. Here you might decode an individual file or open a bigger file to retrieve an individual recording. Be sure to pass decoded events and/or images along to transforms,
  so that they get applied if the users specifies them.

If you are unsure how to do it or need help, just open an issue on Github and don't hesitate to reach out!

Test it
-------
Whether you add a dataset or transform, please make sure to also add a test for it.
Especially the functional tests for transforms are needed to make sure that transformations do what they are supposed to do.
To add your functional test, have a look at the tests already there under `tonic/test`.
To run the tests, execute the following command from the root directory of the repo:
::

  python -m pytest

If you are working on something to try to make a specific test pass, you can also add a filter to save time:
::

  python -m pytest -k my_test_name
