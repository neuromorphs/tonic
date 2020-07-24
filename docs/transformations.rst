Transforms
==========================

.. currentmodule:: tonic.transforms

Transforms are common event transformations. They can be chained together using :class:`Compose`.
Additionally, there is the :mod:`tonic.functional` module.
Functional transforms give fine-grained control over the transformations.
This is useful if you have to build a more complex transformation pipeline.

.. autoclass:: Compose

Transforms on events
-----------------------
.. autoclass:: Crop
.. autoclass:: DropEvents
.. autoclass:: FlipLR
.. autoclass:: FlipPolarity
.. autoclass:: FlipUD
.. autoclass:: MaskIsolated
.. autoclass:: RefractoryPeriod
.. autoclass:: SpatialJitter
.. autoclass:: TimeJitter
.. autoclass:: TimeReversal
.. autoclass:: TimeSkew
.. autoclass:: UniformNoise

Event Representations
---------------------
.. autoclass:: ToRatecodedFrame
.. autoclass:: ToTimesurface
.. autoclass:: ToSparseTensor

Functional transforms in numpy
------------------------------
.. automodule:: tonic.functional
    :members:
    :undoc-members:
