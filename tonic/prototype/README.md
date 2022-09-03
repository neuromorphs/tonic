# Prototype datasets with Torchdata data pipes

[Torchdata](https://pytorch.org/data/beta/index.html) is a prototype library from PyTorch of common modular data loading primitives for easily constructing flexible and performant data pipelines. 

An iterable-style dataset is an instance of a subclass of `IterableDataset` that implements the `__iter__()` protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.

For example, such a dataset, when called `iter(datapipe)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time.

Here's an usage example with NMNIST implemented using datapipes. 


```python
from tonic.prototype.datasets.nmnist import NMNIST

datapipe = NMNIST(root="./data")
events, target = next(iter(datapipe))
events, target
```




    (array([(10, 30,    937, 1), (33, 20,   1030, 1), (12, 27,   1052, 1), ...,
            ( 7, 15, 302706, 1), (26, 11, 303852, 1), (11, 17, 305341, 1)],
           dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]),
     0)



The dataset is fully compatible with Tonic transforms when the `transform` argument is specified:


```python
from tonic.transforms import ToFrame 
t = ToFrame(sensor_size=NMNIST.sensor_size, time_window=10000)
datapipe = NMNIST(root="./data", transform=t)
frames, target = next(iter(datapipe))
frames[0]
```




    array([[[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]], dtype=int16)

One can also apply transforms externally, by using the `Mapper` datapipe from torchdata. Using the `input_col` optional parameter, one can apply the transform to the events (`input_col=0`) or the target (`input_col=1`) or both, by using a transform that acts on the tuple and not specifying `input_col`. This is useful if you want to implement your transform with a simple function and apply it to the dataset. 

```python 
from torchdata.datapipes.iter import Mapper
datapipe = NMNIST(root="./data")
datapipe = Mapper(datapipe, t, input_col=0) # input_col=0 tells Mapper to apply the function t to the entry number 0 of the tuple. 
frames, target = next(iter(datapipe))
frames[0]
```

    array([[[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]], dtype=int16)


Torchdata provides also built-in batchers, which allow to load batches of structured NumPy arrays without transforming them to normal arrays. The iterator returns a list of tuples `(events, target)`.


```python
datapipe = NMNIST(root="./data")
datapipe = datapipe.batch(batch_size=2)
batch = next(iter(datapipe))
batch
```




    [(array([(10, 30,    937, 1), (33, 20,   1030, 1), (12, 27,   1052, 1), ...,
             ( 7, 15, 302706, 1), (26, 11, 303852, 1), (11, 17, 305341, 1)],
            dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]),
      0),
     (array([( 7, 22,   2463, 0), ( 9, 24,   3432, 0), (15, 13,   3641, 0), ...,
             (30, 23, 304620, 1), ( 7, 15, 305649, 0), ( 0, 27, 306232, 1)],
            dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]),
      0)]



Check [Torchdata documentation](https://pytorch.org/data/beta/index.html) to know more about data pipes and what one can do with them. Torchvision seems to be heavily investing on this new format to allow for much more flexibility in datasets, and we plan on following them in order to make Tonic as compatible as possible with standard deep learning frameworks. 

Please, report any bug if you experiment with data pipes. This is _really experimental_ code. 
## Currently supported datasets
- [ ] CIFAR10DVS
- [ ] DAVISDATA
- [ ] DSEC
- [ ] DVSGesture
- [ ] DVSLip
- [ ] MVSEC
- [ ] NCALTECH101
- [x] NMNIST
- [ ] POKERDVS
- [ ] SMNIST
- [x] STMNIST
- [ ] SHD
- [ ] SSC
- [ ] TUMVIE
- [ ] VPR
