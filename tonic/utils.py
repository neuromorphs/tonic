import numpy as np
import tonic.transforms as transforms


def plot_event_grid(events, axis_array=(1, 3), plot_frame_number=False):
    """Plot events accumulated in a voxel grid for visual inspection.

    Args:
        events: event Tensor of shape [num_events, num_event_channels]
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
            "Please install the matplotlib package to plot events. This is an optional"
            " dependency."
        )

    sensor_size_x = int(events["x"].max() + 1)
    sensor_size_y = int(events["y"].max() + 1)
    sensor_size = (sensor_size_x, sensor_size_y)
    
    transform = transforms.Compose(
        [transforms.ToVoxelGrid(sensor_size=sensor_size, n_time_bins=np.product(axis_array))]
    )


    volume = transform(events)
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

    samples_output = []
    targets_output = []

    max_length = max([sample.shape[0] for sample, target in batch])
    for sample, target in batch:
        sample = torch.tensor(sample)
        samples_output.append(
            torch.cat(
                (
                    sample,
                    torch.zeros(max_length - sample.shape[0], *sample.shape[1:]),
                )
            )
        )
        targets_output.append(target)
    return torch.stack(samples_output, 1), torch.tensor(targets_output)
