![tonic](tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![Travis Build Status](https://travis-ci.com/neuromorphs/tonic.svg?branch=master)](https://travis-ci.com/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)

Battling with all the different file formats of publicly available neuromorphic datasets? No more! 
**Tonic** is a tool to facilitate the download, manipulation and loading of event-based/spike-based data. Have a look at the list of [supported datasets](https://tonic.readthedocs.io/en/latest/datasets.html) and [transformations](https://tonic.readthedocs.io/en/latest/transformations.html)!
It's based on PyTorch Vision for an intuitive interface, so that you spend less time worrying about how to read files and more time on things that matter.

## Install
```bash
pip install tonic
```

## Quickstart
```python
import tonic
import tonic.transforms as transforms

transform = transforms.Compose([transforms.Denoise(time_filter=10000),
                                transforms.TimeJitter(std=10),])

testset = tonic.datasets.NMNIST(save_to='./data',
                                train=False,
                                transform=transform)

testloader = tonic.datasets.DataLoader(testset, shuffle=True)

events, target = next(iter(testloader))
```
## Documentation
You can find the full documentation on Tonic  including examples [on this site](https://tonic.readthedocs.io/en/latest/index.html).
