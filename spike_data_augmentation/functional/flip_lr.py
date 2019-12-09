import numpy as np

from .utils import guess_event_ordering_numpy, is_multi_image


def flip_lr_numpy(
    events,
    images=None,
    sensor_size=(346, 260),
    ordering=None,
    flip_probability=0.5,
    multi_image=None,
):
    """Flips events and images in x. Pixels map as:

        x' = width - x

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        images: ndarray of these possible shapes:
                [num_images, height, width, num_channels],
                [height, width, num_channels],
                [num_images, height, width],
                [height, width]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x'
                  to be in the ordering
        flip_probability: probability of performing the flip
        multi_image: Fix whether or not the first dimension of images is num_images

    Returns:
        - every event with x' = sensor_size[1] - 1 - x
        - images flipped in x
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" in ordering

    if images is not None and multi_image is None:
        multi_image = is_multi_image(images, sensor_size)

    if np.random.rand() <= flip_probability:
        if images is not None and multi_image:
            # multiple images NHW or NHWC
            images = images[:, :, ::-1, ...]
        elif images is not None:
            # single image HW or HWC
            images = images[:, ::-1, ...]

        x_loc = ordering.index("x")

        events[:, x_loc] = sensor_size[0] - 1 - events[:, x_loc]

    return events, images
