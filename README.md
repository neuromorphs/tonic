# Telluride Spike Augmentation toolkit
This repository contains a pipeline of data augmentation methods, the effect of which will be tested on various data sets and SOA methods for event- and spike-based data. The goal is to reduce overfitting in learning algorithms by providing implementations of data augmentation methods for event/spike recordings.

## Augementation methods
- left/right flip
- crop random
- time skew, reversal, or jitter
- spatial jitter
- refractory period
- hot/dead pixels
- additive mixing / combining of recordings
- spatial and/or temporal (affine, nonlinear) transforms
- Gaussian noise addition
- Dropout, subsampling, habituation
- polarity flips

## Possible data sets 
- [MVSEC](https://daniilidis-group.github.io/mvsec/)
- [NMNIST](https://www.garrickorchard.com/datasets/n-mnist)
- [EMNIST](https://www.nist.gov/node/1298471/emnist-dataset)
- [NCARS](https://www.prophesee.ai/dataset-n-cars/)
- N-CALTECH 101
- [POKER-DVS](http://www2.imse-cnm.csic.es/caviar/POKERDVS.html)
- [IBM gestures](http://www.research.ibm.com/dvsgesture/)
- NTI Digits
- TIMIT + noise
- audio words

## Algorithms
- [EV-flownet](https://arxiv.org/pdf/1802.06898.pdf)
- [HOTS/HATS](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)
- [SLAYER](https://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)
- ContrastNet
- Extreme Learning Machine, [possible implementation](https://www.ncbi.nlm.nih.gov/pubmed/31251192)
- Flow/Egomotion/Depth
- Sound localisation
- Analog frontend + universal approximator

## Install pre-commit

```
pip install pre-commit
pre-commit install
```

This will install the [black formatter](https://black.readthedocs.io/en/stable/) to a pre-commit hook. When you use ```git add``` you add files to the current commit, then when you run ```git commit``` the black formatter will run BEFORE the commit itself. If it fails the check, the black formatter will format the file and then present it to you to add it into your commit. Simply run ```git add``` on those files again and do the remainder of the commit as normal.

## Run tests

To install pytest

```
pip install pytest
```

To run the tests, from the root directory of the repo

```
python -m pytest test/
```
