"""
==========
CenterCrop
==========
:class:`~tonic.transforms.CenterCrop` crops the events to a central size.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.CenterCrop(sensor_size=nmnist.sensor_size, size=20),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
