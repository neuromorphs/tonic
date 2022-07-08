"""
=============
ToVoxelGrid
=============

This example showcases the ToTimesurface transform with different parameters. 
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tonic

nmnist = tonic.datasets.NMNIST("../../2_tutorials/data", train=False)
events, label = nmnist[0]

####################################
# ToVoxelGrid n_time_bins=20
# ---------------------------

transform = tonic.transforms.ToVoxelGrid(
    sensor_size=nmnist.sensor_size,
    n_time_bins=20,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
