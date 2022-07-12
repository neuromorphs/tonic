"""
===============
RandomCrop
===============
The :class:`~tonic.transforms.RandomCrop` crops the focal plane
to a smaller target size.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.RandomCrop(
            sensor_size=nmnist.sensor_size, target_size=(10, 10)
        ),
        tonic.transforms.ToFrame(
            sensor_size=(10, 10, 2),
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
