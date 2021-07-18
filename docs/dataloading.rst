Dataloading
===========
Tonic as such does not depend on large training frameworks such as PyTorch or TensorFlow to download datasets and access events. This decision was made to provide maximum flexibility while keeping dependencies to a minimum. We can access events and other data that might be provided with a sample simply by indexing the dataset.::

  events, images, target = dataset[100]

In practice however, especially when training a spiking neural network, we want to be able to load data as fast as possible. This is where external dataloaders come into play, which provide support for pre-fetching data, multiple worker threads, batching and other things. Depending on which training framework you are familiar with, you might prefer one or the other to load your data.

PyTorch DataLoader
------------------
Tonic datasets can be passed directly to a :class:`torch.utils.data.DataLoader`. You can find all the supported functionality `in the official documentation <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`_.

.. note::
  When using the PyTorch dataloader, an additional batch dimension will be prepended to your data.

::

    dataset = tonic.datasets.NMNIST(save_to='./data', train=False)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=4, pin_memory=True)

We can then load a single sample ::

    events, target = next(iter(dataloader))

Or loop over all samples available in the dataset ::

    for events, target in iter(dataloader):
        # do something with events
        # do something with targets
