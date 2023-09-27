from typing import Tuple

import numpy as np

import tonic.transforms as transforms


def plot_event_grid(
    events: np.ndarray,
    axis_array: Tuple[int, int] = (1, 3),
    plot_frame_number: bool = False,
):
    """Plot events accumulated as frames equal to the product of axes for visual inspection.

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
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Please install the matplotlib package to plot events. This is an optional"
            " dependency."
        )

    if "y" in events.dtype.names:
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

    else:
        sensor_size_x = int(events["x"].max() + 1)
        frame_transform = transforms.ToFrame(
            sensor_size=(sensor_size_x, 1, 1), n_time_bins=sensor_size_x * 2
        )

        frames = frame_transform(events)
        plt.imshow(frames.squeeze().T)
        plt.xlabel("Time")
        plt.ylabel("Channels")


def plot_animation(frames: np.ndarray, figsize: Tuple[int, int] = (5, 5)):
    """Helper function that animates a tensor of frames of shape (TCHW). If you run this in a
    Jupyter notebook, you can display the animation inline like shown in the example below.

    Parameters:
        frames: numpy array or tensor of shape (TCHW)

    Example:
        >>> import tonic
        >>> nmnist = tonic.datasets.NMNIST(save_to='./data', train=False)
        >>> events, label = nmnist[0]
        >>>
        >>> transform = tonic.transforms.ToFrame(
        >>>     sensor_size=nmnist.sensor_size,
        >>>     time_window=10000,
        >>> )
        >>>
        >>> frames = transform(events)
        >>> animation = tonic.utils.plot_animation(frames)
        >>>
        >>> # Display the animation inline in a Jupyter notebook
        >>> from IPython.display import HTML
        >>> HTML(animation.to_jshtml())

    Returns:
        The animation object. Store this in a variable to keep it from being garbage collected until displayed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        raise ImportError(
            "Please install the matplotlib package to plot events. This is an optional"
            " dependency."
        )
    fig = plt.figure(figsize=figsize)
    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 1:, ...] = frames
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    ax = plt.imshow(frames[0])
    plt.axis("off")

    def animate(frame):
        ax.set_data(frame)
        return ax

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.show()
    return anim
