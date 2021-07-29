Examples
==========================
Here is some example code to get you started on downloading datasets and applying transforms to them:


Downsample events temporally and spatially and convert to sparse tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic
    import tonic.transforms as transforms

    transform = transforms.Compose([transforms.Downsample(time_factor=1e-3, spatial_factor=0.75),
                                    transforms.ToSparseTensor(),])

    testset = tonic.datasets.NMNIST(save_to='./data',
                                    train=True,
                                    transform=transform)

    tensor, target = testset[1000]


Denoise events and transform to time surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic
    import tonic.transforms as transforms

    transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                    transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

    dataset = tonic.datasets.NMNIST(save_to='./data',
                                    train=False,
                                    transform=transform)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, shuffle=True)
    for surfaces, target in iter(dataloader):
        print("{} surfaces for target {}".format(len(surfaces), target))


Load DAVIS data that includes images and IMU measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic

    dataset = tonic.datasets.DAVISDATA(save_to='./data',
                                       recording='shapes_6dof')

    events, imu, images, target = dataset[100]


Load batches of event recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since the package is compatible with PyTorch, we can make use of multithreaded dataloading and shuffling as well as batching.
Using a custom collate function from ``tonic.utils.pad_tensors``, we can retrieve
batches of sparse event tensors by padding shorter tensors like so:
::

    import tonic
    import tonic.transforms as transforms

    transform = tonic.transforms.Compose([
            tonic.transforms.ToSparseTensor(merge_polarities=True),
            ])
    dataset = tonic.datasets.NMNIST(save_to='./data', transform=transform)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
                            batch_size=10,
                            collate_fn=tonic.utils.pad_tensors,
                            shuffle=True)

    tensors, target = next(iter(dataloader))

Plot events in a grid
~~~~~~~~~~~~~~~~~~~~~
For a quick visual check on events, we can plot them in a grid. You can check
the parameters of the function used in :doc:`utils <utils>`.
::

    import tonic
    import tonic.transforms as transforms

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    events, target = dataset[100]

    tonic.utils.plot_event_grid(events, dataset.ordering)
