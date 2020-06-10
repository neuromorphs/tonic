![tonic](tonic-logo-padded.png)
# Telluride Spike Data Augmentation toolkit
This repository contains a pipeline of data augmentation methods, the effect of which will be tested on various data sets and SOA methods for event- and spike-based data. The goal is to reduce overfitting in learning algorithms by providing implementations of data augmentation methods for event/spike recordings.

## Quickstart
In a terminal: clone this repo and install it
```bash
git clone git@github.com:neuromorphs/tonic.git
cd tonic
pip install -e .
```

In a Python file: choose transforms, a data set and whether you want shuffling enabled!
```python
import tonic
import tonic.transforms as transforms

transform = transforms.Compose([transforms.TimeJitter(variance=10),
                                transforms.FlipLR(flip_probability=0.5),
                                transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

testset = tonic.datasets.NMNIST(save_to='./data',
                                                  train=False,
                                                  transform=transform)

testloader = tonic.datasets.Dataloader(testset, shuffle=True)

for surfaces, target in iter(testloader):
    print("{0} surfaces for target {1}".format(len(surfaces), target))
```

## Documentation
To see a list of all transforms and their possible parameters, it is necessary to build documentation locally. Just run the following commands to do that:
```bash
cd docs
make html
firefox _build/html/index.html
```

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

## Algorithms
- [EV-flownet](https://arxiv.org/pdf/1802.06898.pdf)
- [HOTS/HATS](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)
- [SLAYER](https://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)

## Contribute

#### Install pre-commit

```
pip install pre-commit
pre-commit install
```

This will install the [black formatter](https://black.readthedocs.io/en/stable/) to a pre-commit hook. When you use ```git add``` you add files to the current commit, then when you run ```git commit``` the black formatter will run BEFORE the commit itself. If it fails the check, the black formatter will format the file and then present it to you to add it into your commit. Simply run ```git add``` on those files again and do the remainder of the commit as normal.

#### Run tests

To install pytest

```
pip install pytest
```

To run the tests, from the root directory of the repo

```
python -m pytest test/
```
