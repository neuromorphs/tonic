"""
==========
Downsample
==========
The :class:`~tonic.transforms.Downsample` applies
spatial and/or temporal factors to events.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.Downsample(spatial_factor=0.5),
        tonic.transforms.ToFrame(
            sensor_size=(17, 17, 2),
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
