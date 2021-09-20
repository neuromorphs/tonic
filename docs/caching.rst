Dataset Caching
===============

Unfortunately, most neuromorphic datasets are stored in file formats (like aer files) that optimize for disk space and are often are slow to load.
Furthermore, often a single file comprises of several training samples. For instance, a single DVS recording from the `DVSGestures` dataset comprises all the 11 gestures and several repetitions of each.
Similarly if you are training models by converting DVS recording to image frames, a single recording actually corresponds to several image frames that are used in training.
Under these circumstances, it would be better to load just the data that corresponds to the sample as opposed to the entire raw recording.

Caching
-------

To address this issue, Tonic provides a `CachedDataset`. A `CachedDataset` saves individual samples of your dataset to disk in an efficient and convenient format which accelerates dataloading.
This is done by `caching` a sample on first acquisition to be used in subsequent fetches. 
In practice, this means that while your first epoch might be equally slow as before, the following epochs have a much more efficient data loading.


::

    from tonic.datasets import POKERDVS
    from tonic.datasets.cached_dataset import CachedDataset

    # Define/instantiate a dataset, eg. PokerDVS dataset
    dataset = POKERDVS(
        save_to="./data",
        train=False,
        download=True
    )
    dataset = CachedDataset(dataset)
    for data, label in dataset:
        # Use it like a standard dataset
        ...

.. py:currentmodule:: tonic.datasets.cached_dataset

CachedDataset
^^^^^^^^^^^^^

.. autoclass:: CachedDataset