import numpy as np

from .utils import guess_event_ordering_numpy
from .mix_ev_streams import mix_ev_streams_numpy


def uniform_noise_numpy(
    events,
    sensor_size=(346, 260),
    ordering=None,
    scaling_factor_to_micro_sec=1,
    noise_density=1e-8,
):
    """
    Introduces a fixed number of noise depending on sensor size and noise
    density factor, uniformly distributed across the focal plane and in time.

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x', 'y'
                  and 'y' to be in the ordering
        scaling_factor_to_micro_sec: this is a scaling factor to get to micro
                                     seconds from the time resolution used in the event stream,
                                     as the noise time resolution is fixed to 1 micro second.
        noise_density: A noise density of 1 will mean one noise event for every
                       pixel of the sensor size for every micro second.

    Returns:
        events + noise events in one array
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" and "t" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")

    last_timestamp_micro_seconds = events[-1, t_index] * scaling_factor_to_micro_sec
    first_timestamp_micro_seconds = events[0, t_index] * scaling_factor_to_micro_sec

    recording_length_micro_seconds = (
        last_timestamp_micro_seconds - first_timestamp_micro_seconds
    )
    total_number_of_points = recording_length_micro_seconds * np.product(sensor_size)
    number_of_noise_events = int(total_number_of_points * noise_density)

    noise_x = np.random.uniform(0, sensor_size[0], number_of_noise_events).round()
    noise_y = np.random.uniform(0, sensor_size[1], number_of_noise_events).round()
    noise_t = np.random.uniform(
        first_timestamp_micro_seconds,
        last_timestamp_micro_seconds,
        number_of_noise_events,
    )
    noise_p = np.random.choice([-1, 1], number_of_noise_events)

    noise = np.vstack((noise_x, noise_y, noise_t, noise_p)).T
    noise[:, t_index] /= scaling_factor_to_micro_sec

    event_array = (events, noise)

    events, collisions = mix_ev_streams_numpy(
        event_array, sensor_size=sensor_size, ordering=ordering
    )

    return events
