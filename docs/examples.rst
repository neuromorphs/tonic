Examples
==========================
Here is some example code to get you started on downloading datasets and applying transforms to them:

Denoise events and transform to time surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic
    import tonic.transforms as transforms

    transform = transforms.Compose([transforms.Denoise(time_filter=10000),
                                    transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

    dataset = tonic.datasets.NMNIST(save_to='./data',
                                    train=False,
                                    transform=transform)

    dataloader = tonic.datasets.DataLoader(dataset, shuffle=True)
    for surfaces, target in iter(dataloader):
        print("{} surfaces for target {}".format(len(surfaces), target))


Load DAVIS data that includes images and IMU measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic

    dataset = tonic.datasets.DAVISDATA(save_to='./data',
                                       recording='shapes_6dof')

    dataloader = tonic.datasets.DataLoader(dataset, shuffle=False)

    events, imu, images, target = next(iter(dataloader))


Load batches of event recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since the package is built on top of PyTorch, we can make use of multithreaded dataloading
and shuffling as well as batching.
Using a custom collate function from ``tonic.utils.pad_events``, we can retrieve
batches of events by padding shorter event recordings with 0s like so:
::

    import tonic
    import tonic.transforms as transforms

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    dataloader = tonic.datasets.DataLoader(dataset,
                                           batch_size=10,
                                           collate_fn=tonic.utils.pad_events,
                                           shuffle=True)

    events, target = next(iter(dataloader))

Plot events in a grid
~~~~~~~~~~~~~~~~~~~~~
For a quick visual check on events, we can plot them in a grid. You can check
the parameters of the function used in :doc:`utils <utils>`.
::

    import tonic
    import tonic.transforms as transforms

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    dataloader = tonic.datasets.DataLoader(dataset, shuffle=True)

    events, target = next(iter(dataloader))

    tonic.utils.plot_event_grid(events, dataset.ordering)
