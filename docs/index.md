![tonic](../tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Travis Build Status](https://travis-ci.com/neuromorphs/tonic.svg?branch=master)](https://travis-ci.com/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5079802.svg)](https://doi.org/10.5281/zenodo.5079802)

**Download and manipulate neuromorphic datasets fast and easily!**

__Tonic__ provides publicly available event-based vision and audio {doc}`datasets<datasets>` and {doc}`event transformations<auto_examples/index>`. The package is fully compatible with PyTorch Vision / Audio to give you the flexibility that you need. 

### Getting started
* **{doc}`Install Tonic<getting_started/install>`** via pypi or anaconda.
* **{doc}`Run a first example<getting_started/nmnist>`** with this neuromorphic version of the MNIST dataset.

### Tutorials
If you want you can [run them yourself](https://mybinder.org/v2/gh/neuromorphs/tonic/main?labpath=docs%2Ftutorials) in your browser using Binder.
* **{doc}`Load images alongside events<tutorials/davis_data>`** and apply augmentations.
* **{doc}`Learn how to load data fast<tutorials/fast_dataloading>`** using disk-caching.
* **{doc}`Batching when using events<tutorials/batching>`** is straightforward.
* **{doc}`Slice your dataset into smaller chunks<tutorials/slicing>`** if you need to.
* **{doc}`How to work with larger datasets that output multiple data<tutorials/large_datasets>`** for heavy-duty processing.

### How Tos
* **{doc}`Check out these scripts<how-tos/how-tos>`** if you run into a specific problem.

### API reference
* **{doc}`List of neuromorphic datasets<datasets>`**. Vision and audio datasets.
* **{doc}`List of event transformations<auto_examples/index>`**. Event transforms and representations.
* **{doc}Supported file parsers**. For the various file formats out there.

### Reading material
* **{doc}`Introduction to neuromorphic cameras<reading_material/intro-event-cameras>`** if you've never worked with events/spikes.
* **{doc}`Short intro to spiking neural networks<reading_material/intro-snns>`** and how they work with events.
* **{doc}`Links to external spiking neural network simulators<reading_material/intro-snns>`** to train your network.
* **{doc}`Read about design decisions we made<reading_material/intro-snns>`** in Tonic.

### Getting involved
* **{doc}`Contribution guidelines<getting_involved/contribute>`**. Please read this before opening a pull request.
* **{doc}`Communication channels<getting_involved/communication_channels>`** to get in touch.

### About
* **{doc}`About Tonic<about/about>`**. How the project came to life.
* **{doc}`Release notes<about/release_notes>`**. Version changes.

```{toctree}
:hidden:
getting_started/getting_started
auto_examples/index
datasets
tutorials/tutorials
how-tos/how-tos
reading_material/reading_material
getting_involved/getting_involved
about/about
```