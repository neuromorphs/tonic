tonic.datasets
==========================


All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: tonic.datasets

DVS gestures
~~~~~~~~~~~~
.. autoclass:: IBMGesture

NCALTECH 101
~~~~~~~~~~~~
.. autoclass:: NCALTECH101

NCARS
~~~~~
.. autoclass:: NCARS

NMNIST
~~~~~~
.. autoclass:: NMNIST

POKER DVS
~~~~~~~~~
.. autoclass:: POKERDVS
