"""
==========
DropPixel
==========
The :class:`~tonic.transforms.DropPixel` removes all events
that occur at given pixels.

Notice that in the rendering below all x/y pixels from 0 to 17 have been dropped.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../2_tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.DropPixel(
            coordinates=[[x, y] for x in range(17) for y in range(17)]
        ),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
