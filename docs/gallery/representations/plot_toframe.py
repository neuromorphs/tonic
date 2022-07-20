"""
=======
ToFrame
=======

:class:`~tonic.transforms.ToFrame` creates dense frames from events by binning them in different ways.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

####################################
# ToFrame time_window
# -------------------------

transform = tonic.transforms.ToFrame(
    sensor_size=nmnist.sensor_size,
    time_window=10000,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)


####################################
# ToFrame spike_count
# -------------------------

frame_transform = tonic.transforms.ToFrame(
    sensor_size=nmnist.sensor_size,
    event_count=100,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)

####################################
# ToFrame n_time_bins
# ----------------------

frame_transform = tonic.transforms.ToFrame(
    sensor_size=nmnist.sensor_size,
    n_time_bins=20,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
