Contribute
==========

How do I add a new transformation?
----------------------------------
To add a new transform, please consider the following steps:

* Add a new functional transform using one of the supported backends such as numpy or 
  torch in `tonic/functional`. If the transform is simple, this might not be necessary.
* Add a transform wrapper class in `tonic/transforms`. Specify what it does and its
  parameters in a docstring. The wrapper class chooses the functional backend depending on
  parameters or data type.
* Add a test and sufficient assertions to check its validity. Use `pytest.mark.parametrize` 
  to run the test with different parameters.
* Add the transformation to the `api documentation <https://github.com/neuromorphs/tonic/blob/develop/docs/reference/transformations.rst>`_ 
  so that users can find it. 
* Potentially add a plotting script for the `gallery <https://github.com/neuromorphs/tonic/tree/develop/docs/gallery>`_.

Tonic transforms work on events that are represented as structured numpy arrays, since that is
what Tonic datasets output. This has the advantage that events can be indexed easily 
(e.g. events["t"]) and that custom dtypes can be used (e.g. bools for polarity and floats
for timestamps). A transform should always take in events only and return a copy of the 
modified events. 

When implementing the transform, try to make use of existing functional transforms  
or rely on the user to compose them directly. For example, if your transform works on frames 
only, then the user will have to chain a `ToFrame` transform before yours and you can focus 
on your method. 

How can I add a new dataset?
----------------------------
If you want to add a new dataset, there are 2 main questions to consider:

* Where is the dataset hosted?
* What format is it in?

To download a dataset, Tonic makes use of a copy of pytorch vision's tools that can be found 
`here <https://github.com/neuromorphs/tonic/blob/develop/tonic/datasets/download_utils.py>`_.
Cloud providers such as Dropbox, Google Drive, OneDrive etc. are usually not straightforward 
to automatically download from, so if the dataset owner provides multiple mirrors,
prefer a simple REST-like webpage.

The trickier question is the one of file decoding. There exist a variety of file formats that 
events and related data are saved in, including .txt, .aedat, .dat, .es, .mat, .hdf5, .rosbag 
and more. Ordered by degree of easiness to decode a file format, here is one subjective ranking:

#. **hdf5**. This bundles everything in a single file which prevents it from getting messy on 
   your file system and the h5py python package is reliable and a breeze to use. Easy!
#. **npy** and **mat**. These are file formats that can be handled with ubiquitious numpy and 
   scipy packges. Although they might not be the most compact format, they are a good solution 
   due to easiness of decoding.
#. **rosbag**. This is a popular format used by roboticists which bundles everything in a single 
   file like hdf5. The downside is that there is lacking support for a stand-alone python 
   package to decode these files. Tonic builds on the `importRosbag <https://github.com/event-driven-robotics/importRosbag>`_ 
   package which does a good job at decoding, unless your .rosbags are compressed (as of v1.0.3). 
   Plus it might not be able to decode all the topics in there. Good with potential hickups!
#. **aedat**, **dat** and **es**. Every research lab has their own file format because why not! 
   But support to decode those files in a reliable python package is often scarce. The 
   `Loris <https://github.com/neuromorphic-paris/loris>`_ and `aedat <https://github.com/neuromorphicsystems/aedat>`_ 
   packages are trying their best, but might not be maintained at all times... Plus when a dataset 
   is published, there is no reason why it has to be split in single files. Avoid if possible
#. **txt**. Ever tried decoding 1 million events by splitting strings and converting them to floats? 
   No? I did it and I can tell you that it's incredibly slow... Avoid at all costs.

The way a dataset class is organised generally follows this structure:

- The ``__init__`` function takes the necessary parameters. If the dataset has a train/test split, 
  you can have a `train` parameter. Or it might be split into scenes, or maybe no split at all.
  When indexing the files in your dataset folder, you might want to save the path of each file 
  in a list instead of loading them into memory already. That way a file is only loaded later on
  when the dataset is indexed.
- The ``download`` function makes use of the torchvision tools to download and extract archives 
  / files to the right location. Be sure to have md5 hashes ready for the files so that Tonic 
  can check them for consistency.
- The ``__len__`` function provides the number of samples in your dataset.
- ``__getitem__`` is where the interesting stuff happens. Here you might decode an individual 
  file or open a bigger file to retrieve an individual recording. Be sure to pass decoded events 
  and/or other data along to transforms.

Please also add the new dataset to the documentation.

General considerations
----------------------
No matter what code you're planning to contribute, please consider the following things:

* Reach out via our communication channels early on to get help with implementations or design 
  choices. It will prevent you from doing extra work by getting feedback early on.
* use Python's `Black <https://github.com/psf/black>`_ formatter before submitting new code. 
  You might want to use a `pre-commit hook <https://pre-commit.com/>`_ that will automatically check
  for correct formatting before you commit any code. To install it, type the following in the Tonic 
  repository root::

    pip install pre-commit
    pre-commit install

  Then the next time you commit your code, the black formatting check will be run automatically.

* Spell out variable names to `make it easier <https://devblogs.microsoft.com/oldnewthing/20070406-00/?p=27343>`_ 
  for people to understand your code. For example, `y_index` is preferable to `yi` 
  or `compressed_images` to `comp_imgs`. Add a docstring to your transform and explain the 
  function and its parameters.

Test it
-------
Whether you add a dataset or transform, please make sure to also add a test for it.
To add your functional test, have a look at the tests already there under `tonic/test`.
To run the tests, execute the following command from the root directory of the repo:
::

  python -m pytest

If you are working on something to try to make a specific test pass, you can also add a filter to save time:
::

  python -m pytest -k my_test_name

Building the documentation
--------------------------
To build locally, run::

  cd docs/
  make html
  firefox _build/html/index.html

You might want to consider switching tutorial notebook execution off with
`nb_execution_mode = "off"` in conf.py to
prevent notebooks from being run everytime you build the documentation.
