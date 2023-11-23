"""
=================
EventDownsampling
=================
The :class:`~tonic.transforms.EventDownsampling` applies 
spatio-temporal downsampling to events as per the downsampling method chosen.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.Compose(
    [
        tonic.transforms.EventDownsampling(sensor_size=nmnist.sensor_size, 
                                           target_size=(12, 12), 
                                           dt=0.01, 
                                           downsampling_method="differentiator", 
                                           noise_threshold=0, 
                                           differentiator_time_bins=2),
        tonic.transforms.ToFrame(
            sensor_size=(12, 12, 2),
            time_window=10000,
        ),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
