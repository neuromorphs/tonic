"""
=========
ToBinaRep
=========

The :class:`~tonic.transforms.ToBinaRep` creates binary representations of frames.
"""

import tonic
import tonic.transforms as transforms

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = transforms.Compose(
    [
        transforms.ToFrame(
            sensor_size=nmnist.sensor_size,
            n_time_bins=30 * 2,
        ),
        transforms.ToBinaRep(n_frames=30, n_bits=2),
    ]
)

frames = transform(events)

ani = tonic.utils.plot_animation(frames)
