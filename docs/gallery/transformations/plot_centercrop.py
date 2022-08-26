"""
==========
CenterCrop
==========
:class:`~tonic.transforms.CenterCrop` crops the events to a central size.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

cropped_size = (18, 18)
transform = tonic.transforms.Compose(
    [
        tonic.transforms.CenterCrop(sensor_size=nmnist.sensor_size, size=cropped_size),
        tonic.transforms.ToFrame(
            sensor_size=(*cropped_size, 2),
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
