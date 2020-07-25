![tonic](tonic-logo-padded.png)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)

Tonic provides publicly available spike-based datasets and a pipeline of data augmentation methods based on [PyTorch](https://pytorch.org/), which enables multithreaded dataloading and shuffling as well as batching. Have a look at the list of [datasets](https://tonic.readthedocs.io/en/latest/datasets.html) and [transformations](https://tonic.readthedocs.io/en/latest/transformations.html)!

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
You can find the API documentation on the Tonic readthedocs website: https://tonic.readthedocs.io/en/latest/index.html

## Possible data sets (asterix marks currently supported in this package)
- [MVSEC](https://daniilidis-group.github.io/mvsec/)
- [NMNIST](https://www.garrickorchard.com/datasets/n-mnist) (\*)
- [ASL-DVS](https://github.com/PIX2NVS/NVS2Graph)
- [NCARS](https://www.prophesee.ai/dataset-n-cars/)(\*)
- [N-CALTECH 101](https://www.garrickorchard.com/datasets/n-caltech101)(\*)
- [POKER-DVS](http://www2.imse-cnm.csic.es/caviar/POKERDVS.html) (\*)
- [IBM gestures](http://www.research.ibm.com/dvsgesture/) (\*)
- [TI Digits](https://catalog.ldc.upenn.edu/LDC93S10)
- [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)
