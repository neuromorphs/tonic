"""
==================
SpatialJitter
==================
The :class:`~tonic.transforms.SpatialJitter` jitters
x and y coordinates of events.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.SpatialJitter(
            sensor_size=nmnist.sensor_size, var_x=1, var_y=1, clip_outliers=True
        ),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
