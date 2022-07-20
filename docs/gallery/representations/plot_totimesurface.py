"""
=============
ToTimesurface
=============

:class:`~tonic.transforms.ToTimesurface` creates local or global time surfaces for each event.
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]


####################################
# ToTimesurface global
# --------------------------------

transform = tonic.transforms.ToTimesurface(
    sensor_size=nmnist.sensor_size, surface_dimensions=None, tau=100000, decay="exp"
)

frames = transform(events)

# only plot a few of them
ani = tonic.utils.plot_animation(frames[::70])


####################################
# ToTimesurface local
# --------------------------------

transform = tonic.transforms.ToTimesurface(
    sensor_size=nmnist.sensor_size, surface_dimensions=(9, 9), tau=100000, decay="exp"
)

frames = transform(events)

# only plot a few of them
ani = tonic.utils.plot_animation(frames[::70])
