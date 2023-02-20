![tonic](tonic-logo-padded.png)
[![PyPI](https://img.shields.io/pypi/v/tonic)](https://pypi.org/project/tonic/)
[![codecov](https://codecov.io/gh/neuromorphs/tonic/branch/develop/graph/badge.svg?token=Q0BMYGUSZQ)](https://codecov.io/gh/neuromorphs/tonic)
[![Documentation Status](https://readthedocs.org/projects/tonic/badge/?version=latest)](https://tonic.readthedocs.io/en/latest/?badge=latest)
[![contributors](https://img.shields.io/github/contributors-anon/neuromorphs/tonic)](https://github.com/neuromorphs/tonic/pulse)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neuromorphs/tonic/main?labpath=docs%2Ftutorials)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5079802.svg)](https://doi.org/10.5281/zenodo.5079802)
[![Discord](https://img.shields.io/discord/852094154188259338)](https://discord.gg/V6FHBZURkg)

**Tonic** is a tool to facilitate the download, manipulation and loading of event-based/spike-based data. It's like PyTorch Vision but for neuromorphic data!

## Documentation
You can find the full documentation on Tonic [on this site](https://tonic.readthedocs.io/en/latest/index.html).

* [A first example](https://tonic.readthedocs.io/en/latest/getting_started/nmnist.html) to get a feeling for how Tonic works.
* [Run tutorials in your browser](https://mybinder.org/v2/gh/neuromorphs/tonic/main?labpath=docs%2Ftutorials) quick and easy.
* [List of datasets](https://tonic.readthedocs.io/en/main/datasets.html).
* [List of transformations](https://tonic.readthedocs.io/en/main/auto_examples/index.html).
* [About](https://tonic.readthedocs.io/en/latest/about/info.html) this project.
* [Release notes](https://tonic.readthedocs.io/en/latest/about/release_notes.html) on version changes.

## Install
```bash
pip install tonic
```
or (thanks to [@Tobias-Fischer](https://github.com/Tobias-Fischer))
```
conda install -c conda-forge tonic
```
For the latest pre-release on the develop branch that passed the tests:
```
pip install tonic --pre
```
This package has been tested on:

| Linux    | [![](http://github-actions.40ants.com/neuromorphs/tonic/matrix.svg?only=ci.multitest.ubuntu-latest)](https://github.com/neuromorphs/tonic)|
|----------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Windows**  | [![](http://github-actions.40ants.com/neuromorphs/tonic/matrix.svg?only=ci.multitest.windows-2022)](https://github.com/neuromorphs/tonic) |

## Quickstart
If you're looking for a minimal example to run, this is it!

```python
import tonic
import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose(
    [
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=3000),
    ]
)

testset = tonic.datasets.NMNIST(save_to="./data", train=False, transform=transform)

from torch.utils.data import DataLoader

testloader = DataLoader(
    testset,
    batch_size=10,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
)

frames, targets = next(iter(testloader))
```

## Discussion and questions
Have a question about how something works? Ideas for improvement? Feature request? Please get in touch on the #tonic [Discord channel](https://discord.gg/V6FHBZURkg)
 or alternatively here on GitHub via the [Discussions](https://github.com/neuromorphs/tonic/discussions) page!

## Contributing
Please check out the [contributions](https://tonic.readthedocs.io/en/latest/about/contribute.html) page for details.

## Sponsoring
The development of this library is supported by

<tr><td><a href="https://synsense.ai"><img src="https://www.synsense.ai/wp-content/uploads/2022/03/logo-synsense-blue.svg" alt="SynSense" width="200px"/></a></td><td>


## Citation
If you find this package helpful, please consider citing it:

```BibTex
@software{lenz_gregor_2021_5079802,
  author       = {Lenz, Gregor and
                  Chaney, Kenneth and
                  Shrestha, Sumit Bam and
                  Oubari, Omar and
                  Picaud, Serge and
                  Zarrella, Guido},
  title        = {Tonic: event-based datasets and transformations.},
  month        = jul,
  year         = 2021,
  note         = {{Documentation available under 
                   https://tonic.readthedocs.io}},
  publisher    = {Zenodo},
  version      = {0.4.0},
  doi          = {10.5281/zenodo.5079802},
  url          = {https://doi.org/10.5281/zenodo.5079802}
}
```
