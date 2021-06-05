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

DAVIS Event Camera Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DAVISDATA

DVS gestures
~~~~~~~~~~~~
.. autoclass:: DVSGesture

MVSEC
~~~~~
.. autoclass:: MVSEC

NCALTECH 101
~~~~~~~~~~~~
.. autoclass:: NCALTECH101

N-CARS
~~~~~~
.. autoclass:: NCARS

N-MNIST
~~~~~~~
.. autoclass:: NMNIST

N-TIDIGITS
~~~~~~~~~~~
.. autoclass:: NTIDIGITS

NavGesture
~~~~~~~~~~
.. autoclass:: NavGesture

POKER DVS
~~~~~~~~~
.. autoclass:: POKERDVS

Heidelberg Spiking Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Spiking Heidelberg Dataset (SHD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SHD

Spiking Speech Commands (SSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SSC

