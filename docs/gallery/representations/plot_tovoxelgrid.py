"""
=============
ToVoxelGrid
=============

This example showcases the ToTimesurface transform with different parameters. 
"""

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.ToVoxelGrid(
    sensor_size=nmnist.sensor_size,
    n_time_bins=20,
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
