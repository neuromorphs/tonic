Examples
==========================
You can use the pipeline to apply different data transformations. Since the
package is built on top of PyTorch, we can make use of multithreaded dataloading
and shuffling as well as batching.
Here is some example code to get you started:

Denoise events and transform to time surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import tonic
    import tonic.transforms as transforms
    transform = transforms.Compose([transforms.Denoise(time_filter=10000),
                                    transforms.ToTimesurface(surface_dimensions=(7,7), tau=5e3),])

    testset = tonic.datasets.NMNIST(save_to='./data',
                                    train=False,
                                    transform=transform)

    testloader = tonic.datasets.DataLoader(testset, shuffle=True)
    for surfaces, target in iter(testloader):
        print("{} surfaces for target {}".format(len(surfaces), target))

Load batches of event recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using a custom collate function from ``tonic.utils.pad_events``, we can retrieve
batches of events by padding shorter event recordings with 0s like so:
::

    import tonic
    import tonic.transforms as transforms

    testset = tonic.datasets.NMNIST(save_to='./data', train=False)

    testloader = tonic.datasets.DataLoader(testset,
                                           batch_size=10,
                                           collate_fn=tonic.utils.pad_events,
                                           shuffle=True)

    events, target = next(iter(testloader))
