# Welcome to Tonic!


This package provides popular datasets and transformations for spike-based/event-based data. The aim is to deliver all this using an intuitive interface. **Tonic** is modeled after PyTorch Vision without depending on it to give you the flexibility to use other frameworks as well. Spike-based datasets are published in numerous data formats, scattered around the web and containing different data such as events, images, inertial measurement unit readings and many more. **Tonic** tries to streamline the download, manipulation and loading of such data, to give you more time to work on the important things. In addition to downloading datasets, you can also apply custom transforms to events and images to pre-process data before you feed them to your algorithm. We hope that you find it useful!

To get you started, have a look at the [install](getting_started/install.md) and [getting started](getting_started/getting_started.ipynb) pages.

Package reference

```{toctree}
:caption: Introductory Material
:maxdepth: 1

introductory_material/events-introduction
```

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/install
getting_started/getting_started
getting_started/examples
```

```{toctree}
:caption: Advanced Topics
:maxdepth: 1

advanced/caching
advanced/dataloading
advanced/wrapping_own_data

```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/datasets
reference/transformations
reference/utils
```

```{toctree}
:caption: About
:maxdepth: 1

about/about
about/contribute
about/release_notes
```
