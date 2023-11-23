:orphan:

Transformations
===============

.. currentmodule:: tonic.transforms

Transforms are common event transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`tonic.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline.

.. autoclass:: Compose

Transforms on events
--------------------

.. autosummary::
    :toctree: generated/
    :template: class_transform.rst

    CenterCrop
    CropTime
    Denoise
    Decimation
    DropEvent
    DropEventByTime
    DropEventByArea
    EventDrop
    DropPixel
    Downsample
    MergePolarities
    RandomCrop
    RandomFlipLR
    RandomFlipUD
    RandomFlipPolarity
    RandomTimeReversal
    SpatialJitter
    TimeJitter
    RefractoryPeriod
    TimeAlignment
    TimeSkew
    UniformNoise

Event Representations
---------------------

.. autosummary::
    :toctree: generated/
    :template: class_transform.rst

    NumpyAsType
    ToAveragedTimesurface
    ToFrame
    ToSparseTensor
    ToImage
    ToTimesurface
    ToVoxelGrid
    ToBinaRep

Target transforms
-----------------

.. autosummary::
    :toctree: generated/
    :template: class_transform.rst

    ToOneHotEncoding
    Repeat

Functional transforms in numpy
------------------------------

.. currentmodule:: tonic.functional

.. autosummary::
    :toctree: generated/

    crop_numpy
    decimate_numpy
    denoise_numpy
    drop_event_numpy
    drop_by_time_numpy
    drop_by_area_numpy
    drop_pixel_numpy
    drop_pixel_raster
    identify_hot_pixel
    identify_hot_pixel_raster
    refractory_period_numpy
    spatial_jitter_numpy
    time_jitter_numpy
    time_skew_numpy
    to_averaged_timesurface
    to_frame_numpy
    to_timesurface_numpy
    to_voxel_grid_numpy
    to_bina_rep_numpy
