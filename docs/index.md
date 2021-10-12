![tonic](../tonic-logo-padded.png)

[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Travis Build Status](https://travis-ci.com/neuromorphs/tonic.svg?branch=master)](https://travis-ci.com/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5079802.svg)](https://doi.org/10.5281/zenodo.5079802)

**Download, manipulate and load neuromorphic datasets fast and easily!**

__Tonic__ provides publicly available event-based vision and audio datasets and event-based transformations. The package is fully compatible with PyTorch Vision / Audio to give you the flexibility that you need. 

If you have never worked with event-based data, feel free to read through the [introduction to events](getting_started/events-introduction).

To get you started, have a look at the [install](getting_started/install) page and run a [first example](tutorials/nmnist).

For a list of datasets, have a look [here](reference/datasets) and a list of transformations can be found [here](reference/transformations).

```{toctree}
:caption: Getting Started
:hidden:
getting_started/install
getting_started/events-introduction
```

```{toctree}
:caption: Tutorials
:hidden:
tutorials/nmnist
tutorials/davis_data
tutorials/fast_dataloading
tutorials/batching
tutorials/slicing
tutorials/wrapping_own_data
```

```{toctree}
:caption: API reference
:hidden:
reference/data_classes
reference/datasets
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
