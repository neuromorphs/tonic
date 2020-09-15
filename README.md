![tonic](tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)

Tonic provides publicly available spike-based datasets and data transformations based on [PyTorch](https://pytorch.org/).

Have a look at the list of [supported datasets](https://tonic.readthedocs.io/en/latest/datasets.html) and [transformations](https://tonic.readthedocs.io/en/latest/transformations.html)!

## Install
```bash
pip install tonic
```

## Quickstart
```python
import tonic
import tonic.transforms as transforms

transform = transforms.Compose([transforms.Denoise(time_filter=10000),
                                transforms.TimeJitter(variance=10),])

testset = tonic.datasets.NMNIST(save_to='./data',
                                train=False,
                                transform=transform)

testloader = tonic.datasets.DataLoader(testset, shuffle=True)

events, target = next(iter(testloader))
```

## Documentation
You can find the full documentation on Tonic [here](https://tonic.readthedocs.io/en/latest/index.html).
