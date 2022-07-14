"""
===============
DropEventByArea
===============
The :class:`~tonic.transforms.DropEventByArea` removes
all events located in a randomly chosen box area.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.DropEventByArea(
            sensor_size=nmnist.sensor_size, area_ratio=0.4
        ),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
