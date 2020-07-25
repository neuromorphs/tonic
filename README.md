![tonic](tonic-logo-padded.png)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)

Tonic provides publicly available spike-based datasets and a pipeline of data augmentation methods based on [PyTorch](https://pytorch.org/), which enables multithreaded dataloading and shuffling as well as batching.

Have a look at the list of [supported datasets](https://tonic.readthedocs.io/en/latest/datasets.html) and [transformations](https://tonic.readthedocs.io/en/latest/transformations.html)!

## Install
```bash
pip install tonic
```

## Quickstart
```python
import tonic
import tonic.transforms as transforms

transform = transforms.Compose([transforms.TimeJitter(variance=10),
                                transforms.FlipLR(flip_probability=0.5),
                                transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

testset = tonic.datasets.NMNIST(save_to='./data',
                                train=False,
                                transform=transform)

testloader = tonic.datasets.DataLoader(testset, shuffle=True)

for surfaces, target in iter(testloader):
    print("{} surfaces for target {}".format(len(surfaces), target))
```

## Documentation
You can find the full documentation on Tonic [here](https://tonic.readthedocs.io/en/latest/index.html).
