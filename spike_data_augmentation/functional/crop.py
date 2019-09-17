import numpy as np

from .utils import guess_event_ordering_numpy, is_multi_image


def crop_numpy(
    events,
    images=None,
    sensor_size=(346, 260),
    ordering=None,
    target_size=(256, 256),
    multi_image=None,
):
    """Crops the sensor size to a smaller sensor.
    Removes events outsize of the target sensor and maps

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        images: ndarray of these possible shapes:
                - [num_images, height, width, num_channels]
                - [height, width, num_channels]
                - [num_images, height, width]
                - [height, width]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                 the system will take a guess through
                 guess_event_ordering_numpy. This function requires 'x'
                 and 'y' to be in the ordering
        target_size: size of the sensor that was used [W',H']
        multi_image: Fix whether or not the first dimension of images is
                    num_images

    Returns:
        events - events within the crop box
        images - crop box out of the images
    """

    assert target_size[0] <= sensor_size[0] and target_size[1] <= sensor_size[1]

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" in ordering

    if images is not None and multi_image is None:
        multi_image = is_multi_image(images, sensor_size)

    x_start_ind = int(np.random.rand() * (sensor_size[0] - target_size[0]))
    y_start_ind = int(np.random.rand() * (sensor_size[1] - target_size[1]))

    x_end_ind = x_start_ind + target_size[0]
    y_end_ind = y_start_ind + target_size[1]

    if images is not None and multi_image:
        # multiple images NHW or NHWC
        images = images[:, y_start_ind:y_end_ind, x_start_ind:x_end_ind, ...]
    elif images is not None:
        # single image HW or HWC
        images = images[y_start_ind:y_end_ind, x_start_ind:x_end_ind, ...]

    x_loc = ordering.index("x")
    y_loc = ordering.index("y")

    event_mask = (
        (events[:, x_loc] >= x_start_ind)
        * (events[:, x_loc] < x_end_ind)
        * (events[:, y_loc] >= y_start_ind)
        * (events[:, y_loc] < y_end_ind)
    )

    events = events[event_mask, ...]
    events[:, x_loc] -= x_start_ind
    events[:, y_loc] -= y_start_ind

    return events, images
