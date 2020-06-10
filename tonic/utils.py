from matplotlib import animation, rc
import matplotlib.pyplot as plt
import tonic.transforms as transforms

# needs matplotlib widget backend
def plot_events(events, sensor_size, ordering, frame_time=25000, repeat=False):
    from .utils import plot_frames

    transform = transforms.Compose(
        [
            transforms.MaskIsolated(time_filter=20000),
            transforms.ToRatecodedFrame(frame_time=frame_time, merge_polarities=True),
        ]
    )
    frames = transform(events, sensor_size=sensor_size, ordering=ordering)
    plot_frames(frames, frame_time)


def plot_frames(frames, frame_time, repeat=False, vmax=255):
    plt.close()
    fig, ax = plt.subplots()
    axisImage = ax.imshow(frames[0, :, :], cmap=plt.cm.plasma, vmin=0, vmax=vmax)

    def animate(i):
        fig.suptitle("frames {0}".format(i))
        axisImage.set_data(frames[i, :, :])
        return (axisImage,)

    return animation.FuncAnimation(
        fig,
        animate,
        frames=frames.shape[0],
        interval=frame_time / 1000,
        blit=True,
        repeat=repeat,
    )
