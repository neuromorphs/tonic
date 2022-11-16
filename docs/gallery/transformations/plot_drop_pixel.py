"""
==========
DropPixel
==========
The :class:`~tonic.transforms.DropPixel` removes all events
that occur at given pixels.
"""

import numpy as np

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.DropPixel(
            coordinates=[
                [x, y]
                for x in np.random.randint(34, size=29)
                for y in np.random.randint(34, size=29)
            ]
        ),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
