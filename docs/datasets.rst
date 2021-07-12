Datasets
========

All datasets are subclasses of :class:`tonic.datasets.Dataset` and need certain methods implemented: ``__init__``,  ``__getitem__`` and ``__len__``. This design is inspired by torchvision's way to provide datasets.
Even though the arguments that can be passed to a dataset might differ a bit from case to case, they all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and targets respectively.

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

Spiking MNIST
^^^^^^^^^^^^^
.. autoclass:: SMNIST

NavGesture
^^^^^^^^^^
.. autoclass:: NavGesture

POKER DVS
^^^^^^^^^
.. autoclass:: POKERDVS

Visual Place Recognition
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: VPR
