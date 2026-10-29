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
    NTIDIGITS18

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

Object tracking
-------------------
.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    ThreeET_Eyetracking
    NeRDD
    EBSSA

For ``ThreeET_Eyetracking``, the default ``split_version="public_v1"`` is the
official public benchmark: 16 training recordings and 2 validation recordings
with no recording IDs in common. It is recording-disjoint, not subject-disjoint,
and is distinct from the 22-recording experimental subset described in the 3ET
paper. For historical reproduction, ``split_version="legacy_tonic_2023"`` reads
the original archive manifests; those manifests contain ``subj_18_vid_7`` in
both splits and can leak a recording between training and validation. Selecting
that legacy version emits a warning. Both split versions reuse the same raw
archive.
