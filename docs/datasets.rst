Datasets
==========================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    dataloader = tonic.datasets.DataLoader(dataset, shuffle=True, num_workers=4)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: tonic.datasets

ASL-DVS
~~~~~~~
.. autoclass:: ASLDVS

DVS gestures
~~~~~~~~~~~~
.. autoclass:: DVSGesture

NCALTECH 101
~~~~~~~~~~~~
.. autoclass:: NCALTECH101

NCARS
~~~~~
.. autoclass:: NCARS

NMNIST
~~~~~~
.. autoclass:: NMNIST

NavGesture
~~~~~~~~~~
.. autoclass:: NavGesture

POKER DVS
~~~~~~~~~
.. autoclass:: POKERDVS

Dataset suggestions
~~~~~~~~~~~~~~~~~~~~
We would like to include support for some other datasets as well. Some possible candidates are:

* `MVSEC <https://daniilidis-group.github.io/mvsec/>`_
* `TI Digits <https://catalog.ldc.upenn.edu/LDC93S10>`_
* `TIMIT <https://catalog.ldc.upenn.edu/LDC93S1>`_
