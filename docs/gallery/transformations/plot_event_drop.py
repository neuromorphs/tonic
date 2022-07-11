"""
==========
EventDrop
==========
The :class:`~tonic.transforms.EventDrop` removes
events following one of 4 strategies (chosen randomly).
See the related paper: https://arxiv.org/pdf/2106.05836.pdf
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../2_tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.EventDrop(nmnist.sensor_size),
        tonic.transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            n_time_bins=20,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
