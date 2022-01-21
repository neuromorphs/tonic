datasets
========

All datasets are subclasses of :class:`tonic.datasets.Dataset` and need certain methods implemented: ``__init__``,  ``__getitem__`` and ``__len__``. This design is inspired by torchvision's way to provide datasets.
Even though the arguments that can be passed to a dataset might differ a bit from case to case, they all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and targets respectively.

Events for a sample in both audio and vision datasets are output as numpy arrays with shape (N,E), where N is the number of events and E is the number of event channels.

.. currentmodule:: tonic.datasets

Vision datasets
---------------
Vision events typically have 4 event channels: time, x and y pixel coordinates and polarity.

ASL-DVS
^^^^^^^
.. autoclass:: ASLDVS
    :members: __getitem__

Cifar10-DVS
^^^^^^^^^^^
.. autoclass:: CIFAR10DVS
    :members: __getitem__

DAVIS Event Camera Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DAVISDATA
    :members: __getitem__
    
DSEC
^^^^
.. autoclass:: DSEC
    :members: __getitem__

DVS gestures
^^^^^^^^^^^^
.. autoclass:: DVSGesture
    :members: __getitem__

MVSEC
^^^^^
.. autoclass:: MVSEC
    :members: __getitem__

N-CALTECH 101
^^^^^^^^^^^^^
.. autoclass:: NCALTECH101
    :members: __getitem__

N-MNIST
^^^^^^^
.. autoclass:: NMNIST
    :members: __getitem__

POKER DVS
^^^^^^^^^
.. autoclass:: POKERDVS
    :members: __getitem__

Spiking MNIST
^^^^^^^^^^^^^
.. autoclass:: SMNIST
    :members: __getitem__

TUM-VIE
^^^^^^^
.. autoclass:: TUMVIE
    :members: __getitem__

Visual Place Recognition
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: VPR
    :members: __getitem__


Audio datasets
--------------
Audio events typically have 3 event channels: time, frequency channel number and polarity.

Spiking Heidelberg Digits (SHD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SHD

Spiking Speech Commands (SSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SSC

N-TIDIGITS
^^^^^^^^^^
.. autoclass:: NTIDIGITS