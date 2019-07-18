import numpy as np

from .utils import guess_event_ordering_numpy


def uniform_noise_numpy(
    events,
    sensor_size=(346, 260),
    ordering=None,
    noise_temporal_resolution=10,
    time_scaling_factor=1,
    noise_threshold=0.00001,
):
    """
    Introduces noise uniformly distributed across the focal plane and in time.

    Arguments:
    - events - ndarray of shape [num_events, num_event_channels]
    - sensor_size - size of the sensor that was used [W,H]
    - ordering - ordering of the event tuple inside of events, if None
                 the system will take a guess through
                 guess_event_ordering_numpy. This function requires 'x', 'y'
                 and 'y' to be in the ordering
    - noise_temporal_resolution - set the minimal distance in time between
                two noise events in microseconds

    Returns:
    - events - returns events + noise events in one array
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
        assert "x" and "y" and "t" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")

    last_timestamp_micro_seconds = events[-1, t_index] * time_scaling_factor
    noise_probabilities = np.random.randn(
        int(last_timestamp_micro_seconds / noise_temporal_resolution),
        sensor_size[0],
        sensor_size[1],
    )
    noise_positive = np.transpose(
        np.where(
            np.logical_and(
                noise_probabilities > 0, noise_probabilities < noise_threshold
            )
        )
    )
    noise_negative = np.transpose(
        np.where(
            np.logical_and(
                noise_probabilities < 0, noise_probabilities > (-noise_threshold)
            )
        )
    )

    noise_positive = np.append(
        noise_positive, np.ones((len(noise_positive), 1)), axis=1
    )
    noise_negative = np.append(
        noise_negative, -(np.ones((len(noise_negative), 1))), axis=1
    )
    print(noise_negative)
    print(last_timestamp_micro_seconds)
    print("len: " + str(len(noise_positive)))
    print("len: " + str(len(noise_negative)))

    return events
