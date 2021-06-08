Contribute
==========

How can I add a new dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to add a new dataset, there are 2 main questions to consider:

  * Where is the dataset hosted?
  * What format is it in?

To download a dataset, Tonic makes use of pytorch vision's tools that can be found `here <https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py>`_.
Cloud providers such as Dropbox, Google Drive, OneDrive etc. are usually a pain to automatically download from, so if the dataset owner provides multiple mirrors,
prefer a simple REST-like webpage such as https://www.neuromorphic-vision.com/public/downloads/.


The trickier question is the dataset decoding. There exist a variety of file formats that events and related data are saved in, including .txt, .aedat, .dat, .es, .mat, .hdf5, .rosbag and more.
Ordered by degree of easiness to decode a file format, here is one subjective ranking:

  1. **hdf5**. This bundles everything in a single file which prevents it from getting messy on your file system and the h5py python package is reliable and a breeze to use.
  Easy!

  2. **rosbag**. This is a popular format used by roboticists which similarly bundles everything in a single file. The downside however that there is lacking support
  for a stand-alone python package to decode these files. Tonic builds on the `importRosbag <https://github.com/event-driven-robotics/importRosbag>`_ package which
  does a great job at decoding, unless your .rosbags are compressed. Plus it might not be able to decode all the topics in there. Good with potential hickups!

  3. **aedat**, **dat**, **mat** and **es**. Every lab has their own file format because why not! But support to decode those files in a reliable python package is often scarce.
  The `Loris <https://github.com/neuromorphic-paris/loris>`_ and `aedat <https://github.com/neuromorphicsystems/aedat>`_ packages are trying their best, but might
  not be maintained at all times... Plus when a dataset is published, there is no reason why it has to be split in single files. Avoid if possible

  4. **txt**. Ever tried decoding 1 million events by splitting strings and converting them to floats? No?
  Keep it that way...

The way a dataset class is organised generally follows this structure:

  - The ``__init__`` function takes the necessary parameters. If the dataset has a train/test split, you can have a `train` parameter. Or might be split into scenes, or maybe no split at all.
    If the dataset consists of many individual files, you can save the path to each file, so that you don't have to read the whole dataset into memory at once.
  - The ``download`` function makes use of the torchvision tools to download and extract archives / files to the right location. Be sure to have md5 hashes ready for the files so that Tonic can check them for conistency.
  - The ``__len__`` function provides the number of samples in your dataset.
  - ``__getitem__`` is where the interesting stuff happens. Here you might decode an individual file or open a bigger file to retrieve an individual recording. Be sure to pass events and/or images along to transforms,
    so that they get applied if the users specifies them.

If you are unsure how to do it or need help, just open an issue on Github and don't hesitate to reach out!

Test it
~~~~~~~~~
Whether you add a dataset or transform, please make sure to also add a test for it.
Especially the functional tests for transforms are needed to make sure that transformations do what they are supposed to do.
To add your functional test, have a look at the tests already there under `tonic/test`.
To run the tests, execute the following command from the root directory of the repo:
::

  python -m pytest

If you are working on something to try to make a specific test pass, you can also add a filter to save time:
::

  python -m pytest -k my_test_name
