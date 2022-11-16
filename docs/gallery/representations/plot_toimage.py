"""
=======
ToImage
=======

:class:`~tonic.transforms.ToImage` bins all the events to a single image.
Typically used in conjunction with SlicedDataset, to chunk recordings
into smaller samples.

"""

import matplotlib.pyplot as plt

import tonic

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.ToImage(
    sensor_size=nmnist.sensor_size,
)

image = transform(events[:1000])

plt.imshow(image[1] - image[0])
plt.axis(False)
plt.show()
