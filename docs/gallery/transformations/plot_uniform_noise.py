"""
==================
UniformNoise
==================
The :class:`~tonic.transforms.UniformNoise` transform
adds a specific number of noise events to the sample.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.UniformNoise(sensor_size=nmnist.sensor_size, n=1000),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
