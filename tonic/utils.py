import numpy as np
import tonic.transforms as transforms


def plot_event_grid(events, ordering, axis_array=(1, 3), plot_frame_number=False):
    """Plot events accumulated in a voxel grid for visual inspection.

    Args:
        events: event Tensor of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events,
                    for example 'xytp'.
        axis_array: dimensions of plotting grid. The larger the grid,
                    the more fine-grained the events will be sliced in time.
        plot_frame_number: optional index of frame when plotting

    Returns:
        None
    """
    try:
        from matplotlib import animation, rc
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Please install the matplotlib package to plot events. This is an optional dependency."
        )

    events = events.squeeze()
    events = np.array(events)
    transform = transforms.Compose(
        [transforms.ToVoxelGrid(num_time_bins=np.product(axis_array))]
    )
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    sensor_size_x = int(events[:, x_index].max() + 1)
    sensor_size_y = int(events[:, y_index].max() + 1)
    sensor_size = (sensor_size_x, sensor_size_y)

    volume = transform(events, sensor_size=sensor_size, ordering=ordering)
    fig, axes_array = plt.subplots(*axis_array)

    if 1 in axis_array:
        for i in range(np.product(axis_array)):
            axes_array[i].imshow(volume[i, :, :])
            axes_array[i].axis("off")
            if plot_frame_number:
                axes_array[i].title.set_text(str(i))
    else:
        for i in range(axis_array[0]):
            for j in range(axis_array[1]):
                axes_array[i, j].imshow(volume[i * axis_array[1] + j, :, :])
                axes_array[i, j].axis("off")
                if plot_frame_number:
                    axes_array[i, j].title.set_text(str(i * axis_array[1] + j))
    plt.tight_layout()
    plt.show()


def pad_tensors(batch):
    """This is a custom collate function for a pytorch dataloader to load multiple
    event recordings at once. It's intended to be used in combination with sparse tensors.
    All tensor sizes are extended to the largest one in the batch, i.e. the longest recording.

    Example:
        >>> dataloader = torch.utils.data.DataLoader(dataset,
        >>>                                          batch_size=10,
        >>>                                          collate_fn=tonic.utils.pad_tensors,
        >>>                                          shuffle=True)

    """
    import torch

    if not isinstance(batch[0][0], torch.Tensor):
        print(
            "tonic.utils.pad_tensors expects a PyTorch Tensor of events. Please use ToSparseTensor or similar transform to convert the events."
        )
        return None, None
    max_length = max([sample.size()[0] for sample, target in batch])

    samples_output = []
    targets_output = []
    for sample, target in batch:
        sample.sparse_resize_(
            (max_length, *sample.size()[1:]), sample.sparse_dim(), sample.dense_dim()
        )
        samples_output.append(sample)
        targets_output.append(target)
    return torch.stack(samples_output), targets_output
