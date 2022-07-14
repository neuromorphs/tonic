"""
=======
ToImage
=======

"""

import tonic
import matplotlib.pyplot as plt

nmnist = tonic.datasets.NMNIST("../../tutorials/data", train=False)
events, label = nmnist[0]

transform = tonic.transforms.ToImage(
    sensor_size=nmnist.sensor_size,
)

image = transform(events)

plt.imshow(image[1] - image[0])
plt.axis(False)
plt.show()
