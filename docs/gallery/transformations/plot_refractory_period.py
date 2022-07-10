"""
==================
RefractoryPeriod
==================
The :class:`~tonic.transforms.RefractoryPeriod` drops 
all events during the refractory period.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.RefractoryPeriod(refractory_period=10000),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
