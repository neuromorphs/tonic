import numpy as np
from matplotlib import animation, rc
import matplotlib.pyplot as plt
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


def pad_events(batch):
    """This is a custom collate function for a pytorch dataloader to load multiple
    event samples at once and pad the recordings with the fewest events.

    Example:
        >>> dataloader = tonic.datasets.DataLoader(dataset,
        >>>                                        batch_size=10,
        >>>                                        collate_fn=tonic.utils.pad_events,
        >>>                                        shuffle=True)

    """
    max_length = 0
    for sample, target in batch:
        if len(sample) > max_length:
            max_length = len(sample)
    samples_output = []
    targets_output = []
    for sample, target in batch:
        sample = np.vstack((np.zeros((max_length - len(sample), 4)), sample))
        samples_output.append(sample)
        targets_output.append(target)
    return np.stack(samples_output), targets_output
