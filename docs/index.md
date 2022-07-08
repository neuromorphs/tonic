![tonic](../tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Travis Build Status](https://travis-ci.com/neuromorphs/tonic.svg?branch=master)](https://travis-ci.com/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5079802.svg)](https://doi.org/10.5281/zenodo.5079802)

**Download and manipulate neuromorphic datasets fast and easily!**

__Tonic__ provides publicly available event-based vision and audio datasets and event transformations. The package is fully compatible with PyTorch Vision / Audio to give you the flexibility that you need. 

### Getting started
* **{doc}`Install Tonic<1_getting_started/install>`** via pypi or anaconda.
* **{doc}`Run a first example<1_getting_started/nmnist>`** with this neuromorphic version of the MNIST dataset.

### Tutorials
If you want you can [run them yourself](https://mybinder.org/v2/gh/neuromorphs/tonic/main?labpath=docs%2Ftutorials) in your browser using Binder.
* **{doc}`Load images alongside events<2_tutorials/davis_data>`** and apply augmentations.
* **{doc}`Learn how to load data fast<2_tutorials/fast_dataloading>`** using disk-caching.
* **{doc}`Batching when using events<2_tutorials/batching>`** is straightforward.
* **{doc}`Slice your dataset into smaller chunks<2_tutorials/slicing>`** if you need to.
* **{doc}`How to work with larger datasets that output multiple data<2_tutorials/large_datasets>`** for heavy-duty processing.
* **{doc}`If you have your own data<2_tutorials/slicing>`** you can still use Tonic.

### How Tos
* **{doc}`Troubleshoot<3_how-tos/how-tos>`** your stack traces.

### API reference
* **{doc}`List of neuromorphic datasets<4_reference/datasets>`**. Vision and audio datasets.
* **{doc}`List of event transformations<4_reference/transformations>`**. Event transforms and representations.
* **{doc}`Supported file parsers<4_reference/io>`**. For the various file formats out there.

### Reading material
* **{doc}`Introduction to neuromorphic cameras<5_reading_material/intro-event-cameras>`** if you've never worked with events/spikes.
* **{doc}`Short intro to spiking neural networks<5_reading_material/intro-snns>`** and how they work with events.
* **{doc}`Links to external spiking neural network simulators<5_reading_material/intro-snns>`** to train your network.

### Getting involved
* **{doc}`Contribution guidelines<6_getting_involved/contribute>`**. Please read this before opening a pull request.

### About
* **{doc}`About Tonic<7_about/about>`**. How the project came to life.
* **{doc}`Release notes<7_about/release_notes>`**. Version changes.

```{toctree}
:hidden:
1_getting_started/getting_started
auto_examples/index
2_tutorials/tutorials
3_how-tos/how-tos
4_reference/reference
5_reading_material/reading_material
6_getting_involved/getting_involved
7_about/about
```
