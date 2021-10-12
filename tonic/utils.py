import numpy as np
import tonic.transforms as transforms


def plot_event_grid(events, axis_array=(1, 3), plot_frame_number=False):
    """Plot events accumulated in a voxel grid for visual inspection.

    Parameters:
        events: Structured numpy array of shape [num_events, num_event_channels].
        axis_array: dimensions of plotting grid. The larger the grid,
                    the more fine-grained the events will be sliced in time.
        plot_frame_number: optional index of frame when plotting

    Example:
        >>> import tonic
        >>> dataset = tonic.datasets.NMNIST(save_to='./data')
        >>> events, target = dataset[100]
        >>> tonic.utils.plot_event_grid(events)

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
    sensor_size_p = len(np.unique(events["p"]))
    sensor_size = (sensor_size_x, sensor_size_y, sensor_size_p)

    transform = transforms.ToFrame(
        sensor_size=sensor_size, n_time_bins=np.product(axis_array)
    )

    frames = transform(events)
    fig, axes_array = plt.subplots(*axis_array)

    if 1 in axis_array:
        axes_array = axes_array.reshape(1, -1)

    for i in range(axis_array[0]):
        for j in range(axis_array[1]):
            frame = frames[i * axis_array[1] + j]
            axes_array[i, j].imshow(frame[1] - frame[0])
            axes_array[i, j].axis("off")
            if plot_frame_number:
                axes_array[i, j].title.set_text(str(i * axis_array[1] + j))
    plt.tight_layout()
    plt.show()
