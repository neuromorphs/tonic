"""
===============
MergePolarities
===============
The :class:`~tonic.transforms.MergePolarities` sets all polarities to zero.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.MergePolarities(),
        tonic.transforms.ToFrame(
            sensor_size=(34, 34, 1),
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
