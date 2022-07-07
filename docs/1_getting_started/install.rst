Install
=======

The latest stable releases of **Tonic** are available on pypi, so you can just type::

  pip install tonic

**Tonic** is also available via conda! If you prefer to use conda, please follow the instructions as `outlined here <https://github.com/conda-forge/tonic-feedstock>`_.

If you'd like to use the latest pre-release, which is the latest code on the develop branch that passed the tests, you can also install::

  pip install tonic --pre

Requirements
------------
Even though Tonic is modeled after PyTorch Vision, it does not depend on it or PyTorch for that matter. We want to keep our package lightweight to provide a minimum set of functionality, while at the same time being compatible with mature dataloading classes in larger frameworks. We keep dependencies to a minimum set of packages that help with decoding different file formats such as hd5f, rosbags and others.

Supported platforms
--------------------
Tonic being a Python only package, support across platforms should be without issues for the most part. We test Tonic on the earliest and latest Python 3 version that is officially supported, both on Linux and Windows. If you run into any problems installing it on your system, please let us know.
