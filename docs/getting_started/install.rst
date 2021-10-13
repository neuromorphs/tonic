Install
=======

The latest releases of **Tonic** are available on pypi, so you can just type
::

  pip install tonic

**Tonic** is also available via conda! If you prefer to use conda, please follow
the instructions as `outlined here <https://github.com/conda-forge/tonic-feedstock>`_.

Requirements
------------
Even though Tonic is modeled after PyTorch Vision, it does not depend on it or PyTorch for that matter. We want to keep our package lightweight to provide a minimum set of functionality, while at the same time being compatible with mature dataloading classes in larger frameworks. We keep dependencies to a minimum set of packages that help with decoding different file formats such as hd5f, rosbags and others.

Supported platforms
--------------------
Tonic being a Python only package, support across platforms should be without issues for the most part. If you run into any problems installing it on your system, please open a Github issue and we'll look into it.
