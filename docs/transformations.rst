Transforms
==========================

.. currentmodule:: tonic.transforms

Transforms are common event transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`tonic.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline.

.. autoclass:: Compose

Transforms on events
--------------------
Cropping
^^^^^^^^
.. autoclass:: Crop

Denoising
^^^^^^^^^
.. autoclass:: Denoise

Randomly dropping events
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropEvents

Flipping left/right and up/down
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: FlipLR
.. autoclass:: FlipUD

Flipping polarities
^^^^^^^^^^^^^^^^^^^
.. autoclass:: FlipPolarity

Masking hot pixels that fire excessively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaskHotPixel

Refractory periods
^^^^^^^^^^^^^^^^^^
.. autoclass:: RefractoryPeriod

Jittering events spatially
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialJitter

Jittering events temporally
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TimeJitter

Reverse time
^^^^^^^^^^^^
.. autoclass:: TimeReversal

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

Rate-coded frames
^^^^^^^^^^^^^^^^^
.. autoclass:: ToRatecodedFrame

Sparse tensor
^^^^^^^^^^^^^
.. autoclass:: ToSparseTensor

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
