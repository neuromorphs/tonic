import numpy as np

import warnings

from .utils import guess_event_ordering_numpy, is_multi_image


def time_reversal_numpy(
    events,
    images=None,
    sensor_size=(346, 260),
    ordering=None,
    flip_probability=0.5,
    multi_image=None,
):
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

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
        - every event flipped in time and polarity or no change and
        - images flipped in time
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "t" and "p" in ordering

    if images is not None and multi_image is None:
        multi_image = is_multi_image(images, sensor_size)

    if multi_image is not None and not multi_image:
        warnings.warn("Time reversal with single image can lead to incorrect results.")

    if np.random.rand() < flip_probability:
        if images is not None and multi_image:
            # multiple images NHW or NHWC
            images = images[::-1, ...]

        t_loc = ordering.index("t")
        p_loc = ordering.index("p")

        events[:, t_loc] = np.max(events[:, t_loc]) - events[:, t_loc]
        events[:, p_loc] *= -1

    return events, images
