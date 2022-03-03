transforms
============

.. currentmodule:: tonic.transforms

Transforms are common event transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`tonic.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline.

.. autoclass:: Compose

Transforms on events
--------------------


Denoising
^^^^^^^^^
.. autoclass:: Denoise

Drop events randomly
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropEvent

Drop (hot) pixels
^^^^^^^^^^^^^^^^^
.. autoclass:: DropPixel

Downsample timestamps and/or spatial coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Downsample

Merge polarities
^^^^^^^^^^^^^^^^^
.. autoclass:: MergePolarities

Random crop
^^^^^^^^^^^^
.. autoclass:: RandomCrop

Randomly flip left/right and up/down
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomFlipLR
.. autoclass:: RandomFlipUD

Randomly flip polarities
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomFlipPolarity

Randomly reverse time
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RandomTimeReversal

Jitter events spatially
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialJitter

Jitter events temporally
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeJitter

Convert data type
^^^^^^^^^^^^^^^^^
.. autoclass:: NumpyAsType

Refractory periods
^^^^^^^^^^^^^^^^^^
.. autoclass:: RefractoryPeriod

Align at time zero
^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeAlignment

Transform time
^^^^^^^^^^^^^^
.. autoclass:: TimeSkew

Add uniform noise
^^^^^^^^^^^^^^^^^
.. autoclass:: UniformNoise


Event Representations
---------------------

Averaged time surfaces
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ToAveragedTimesurface

Frames
^^^^^^
.. autoclass:: ToFrame

Single image
^^^^^^^^^^^^
.. autoclass:: ToImage

Time surfaces
^^^^^^^^^^^^^
.. autoclass:: ToTimesurface

Voxel grid
^^^^^^^^^^^
.. autoclass:: ToVoxelGrid


Target transforms
-----------------
One hot encoding
^^^^^^^^^^^^^^^^^
.. autoclass:: ToOneHotEncoding

Repeat pattern n times
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Repeat


Functional transforms in numpy
------------------------------
.. automodule:: tonic.functional
    :members:
    :undoc-members:
