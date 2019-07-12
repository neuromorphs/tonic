# Data augmentation for event/spike recordings
This repository will contain a pipeline of data augmentation methods, the effect of which will be tested on various data sets and SOA methods for event- and spike-based data.

## Augementation methods
- left/right flip
- crop random
- timestamp jitter
- refractory period, subsampling
- hot/dead pixels
- x, y, t (nonlinear) transforms
- Gaussian noise addition
- Dropout
- polarity flips

## Possible data sets 
- MVSEC
- NMNIST
- NCARS
- N-CALTECH 101
- pip cards
- IBM gestures
- NTI Digits
- audio words
- NTI digits
- TIMIT + noise

## Algorithms
- [EV-flownet](https://arxiv.org/pdf/1802.06898.pdf)
- [HOTS/HATS](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf)
- [SLAYER](https://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)
- ContrastNet
- ELM
- Flow/Egomotion/Depth
- Sound localisation
- Analog frontend + universal approximator
