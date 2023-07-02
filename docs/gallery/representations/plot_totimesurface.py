"""
=============
ToTimesurface
=============

:class:`~tonic.transforms.ToTimesurface` creates global time surfaces at a specific time interval dt.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.ToTimesurface(
    sensor_size=nmnist.sensor_size,
    tau=30000,
    dt=10000,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
