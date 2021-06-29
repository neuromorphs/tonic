Datasets
========

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    dataloader = tonic.datasets.DataLoader(dataset, shuffle=True, num_workers=4)

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.

Events for a sample in both audio and vision datasets are output as numpy arrays with shape (N,E), where N is the number of events and E is the number of event channels. 

.. currentmodule:: tonic.datasets

Audio datasets
--------------
Audio events typically have 3 event channels: time, frequency channel number and polarity.

Heidelberg Spiking Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spiking Heidelberg Dataset (SHD)
""""""""""""""""""""""""""""""""
.. autoclass:: SHD

Spiking Speech Commands (SSC)
"""""""""""""""""""""""""""""
.. autoclass:: SSC

N-TIDIGITS
^^^^^^^^^^
.. autoclass:: NTIDIGITS

Vision datasets
---------------
Vision events typically have 4 event channels: time, x and y pixel coordinates and polarity.

ASL-DVS
^^^^^^^
.. autoclass:: ASLDVS

DAVIS Event Camera Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DAVISDATA

DVS gestures
^^^^^^^^^^^^
.. autoclass:: DVSGesture

MVSEC
^^^^^
.. autoclass:: MVSEC

N-CALTECH 101
^^^^^^^^^^^^^
.. autoclass:: NCALTECH101

N-CARS
^^^^^^
.. autoclass:: NCARS

N-MNIST
^^^^^^^
.. autoclass:: NMNIST

NavGesture
^^^^^^^^^^
.. autoclass:: NavGesture

POKER DVS
^^^^^^^^^
.. autoclass:: POKERDVS

Visual Place Recognition
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: VPR
