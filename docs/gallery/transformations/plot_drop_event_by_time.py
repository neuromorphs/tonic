"""
==========
DropEventByTime
==========
The :class:`~tonic.transforms.DropEventByTime` removes
all events in a certain time interval of a specified 
duration ratio.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../2_tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.DropEventByTime(duration_ratio=0.4),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            n_time_bins=20,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
