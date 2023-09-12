Datasets
========

All datasets are subclasses of :class:`tonic.datasets.Dataset` and need certain methods implemented: ``__init__``,  ``__getitem__`` and ``__len__``. This design is inspired by torchvision's way to provide datasets.

Events for a sample in both audio and vision datasets are output as structured numpy arrays of shape (N,E), where N is the number of events and E is the number of event channels. Vision events typically have 4 event channels: time, x and y pixel coordinates and polarity, whereas audio events typically have time, x and polarity.

.. currentmodule:: tonic.datasets

Visual event stream classification
----------------------------------

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    ASLDVS
    CIFAR10DVS
    DVSGesture
    NCALTECH101
    NMNIST
    POKERDVS
    SMNIST
    DVSLip

Audio event stream classification
---------------------------------

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    SHD
    SSC

Pose estimation, visual odometry, SLAM
--------------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    DAVISDATA
    DSEC
    MVSEC
    TUMVIE
    VPR
    
Star tracking
-------------------
.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    EBSSA

.. currentmodule:: tonic.prototype.datasets

Prototype iterable datasets
---------------------------
.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    NMNIST
    NCARS
    STMNIST
    Gen1AutomotiveDetection
    Gen4AutomotiveDetectionMini
