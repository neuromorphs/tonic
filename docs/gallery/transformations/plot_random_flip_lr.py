"""
===============
RandomFlipLR
===============
The :class:`~tonic.transforms.RandomFlipLR` flips 
events on the horizontal axis with probability p.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.RandomFlipLR(sensor_size=nmnist.sensor_size, p=1),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
