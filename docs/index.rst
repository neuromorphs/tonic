.. Telluride Tonic documentation master file, created by
   sphinx-quickstart on Thu Sep 12 10:54:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tonic!
=======================================================

This package provides popular datasets and transformations for spike-based/event-based data. The aim is to deliver all this using an intuitive interface.
**Tonic** is modeled after PyTorch Vision, so if you're already familiar with that one, you'll find yourself at home right away.
Spike-based datasets are published in numerous data formats, scattered around the web and containing different data such as events, images, intertial measurement unit readings and many more.
**Tonic** tries to streamline the download and manipulation of such data, so that you spend less time worrying about how to read files and more time on things that matter.
In addition to downloading datasets, you can also apply custom transforms to events and images to pre-process data before you feed them to your algorithm. We hope that you find it useful!

Install
-------
**Tonic** is available on pypi, so just type
::

  pip install tonic

If you run into any problems installing it on your system, please open a Github issue.

Package reference
-----------------
.. toctree::
   :maxdepth: 2

   examples
   transformations
   datasets
   utils
   contribute
