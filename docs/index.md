![tonic](../tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Travis Build Status](https://travis-ci.com/neuromorphs/tonic.svg?branch=master)](https://travis-ci.com/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5079802.svg)](https://doi.org/10.5281/zenodo.5079802)

**Download and manipulate neuromorphic datasets fast and easily!**

__Tonic__ provides publicly available event-based vision and audio datasets and event transformations. The package is fully compatible with PyTorch Vision / Audio to give you the flexibility that you need. 

### Getting started
* **{doc}`Install Tonic<getting_started/install>`** via pypi or anaconda.
* **{doc}`Introduction to neuromorphic cameras<getting_started/intro-event-cameras>`** if you've never worked with events/spikes.
* **{doc}`Short intro to spiking neural networks<getting_started/intro-snns>`** and how they work with events.
* **{doc}`Links to external spiking neural network simulators<getting_started/intro-snns>`** to train your network.

### Tutorials
If you want you can [run them yourself](https://mybinder.org/v2/gh/neuromorphs/tonic/main?labpath=docs%2Ftutorials) in your browser using Binder.
* **{doc}`Run a first example<tutorials/nmnist>`** with this neuromorphic version of the MNIST dataset.
* **{doc}`Load images alongside events<tutorials/davis_data>`** and apply augmentations.
* **{doc}`Learn how to load data fast<tutorials/fast_dataloading>`** using disk-caching.
* **{doc}`Batching when using events<tutorials/batching>`** is straightforward.
* **{doc}`Slice your dataset into smaller chunks<tutorials/slicing>`** if you need to.
* **{doc}`How to work with larger datasets that output multiple data<tutorials/large_datasets>`** for heavy-duty processing.
* **{doc}`If you have your own data<tutorials/slicing>`** you can still use Tonic.

### API reference
* **{doc}`List of neuromorphic datasets<reference/datasets>`**. Vision and audio datasets.
* **{doc}`List of event transformations<reference/transformations>`**. Event transforms and representations.
* **{doc}`Supported file parsers<reference/io>`**. For the various file formats out there.

### About
* **{doc}`About Tonic<about/about>`**. How the project came to life.
* **{doc}`Contribution guidelines<about/contribute>`**. Please read this before opening a pull request.
* **{doc}`Release notes<about/release_notes>`**. Version changes.


```{toctree}
:caption: Getting Started
:hidden:
getting_started/install
getting_started/intro-event-cameras
getting_started/intro-snns
getting_started/training_snn
```

```{toctree}
:caption: Tutorials
:hidden:
tutorials/nmnist
tutorials/davis_data
tutorials/fast_dataloading
tutorials/large_datasets
tutorials/batching
tutorials/slicing
tutorials/wrapping_own_data
```

```{toctree}
:caption: API reference
:hidden:
reference/data_classes
reference/datasets
reference/collation
reference/io
reference/slicers
reference/transformations
reference/utils
```

```{toctree}
:hidden:
:caption: About
about/about
about/contribute
about/release_notes
```
