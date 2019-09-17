import numpy as np

from .utils import guess_event_ordering_numpy


def mix_ev_streams_numpy(
    events, offsets=None, check_conflicts=False, sensor_size=(346, 260), ordering=None
):

    """Combine two or more event streams into a single stream. Event collisions result in a single spike
    or none if polarities are opposite. Collisions numbering greater than two are handled by consensus.
    While not technically required, it is recommended that all event streams be the same [x,y] dimension.

    Args:
        events: tuple of event streams which are ndarrays of shape [num_events, num_event_channels]
        offsets: tuple of start time offsets for each event stream:
                    - Default all streams start at the same time
                    - Random : applies a random offset from 0 to the timespan of the longest event stream
        check_conflicts: bool, whether or not to check for event collisions. Slower processing if True and probably uneccessary most of the time
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None the system
                  will take a guess through guess_event_ordering_numpy.

    Returns:
        a combined event stream
    """

    if ordering is None:
        assert len(events) > 1
        prime_ordering = guess_event_ordering_numpy(events[0])
        orderings = [guess_event_ordering_numpy(x) for x in events]

        for ordering in orderings:
            assert ordering == prime_ordering

        ordering = prime_ordering

    assert "x" and "y" and "t" in ordering

    x_loc = ordering.find("x")
    y_loc = ordering.find("y")
    t_loc = ordering.find("t")
    p_loc = ordering.find("p")

    # shift events to zero
    events = np.array(events)

    if events.dtype != np.object:
        # event streams are all of same size
        events[:, :, t_loc] -= np.tile(events[:, 0, t_loc], (events.shape[1], 1)).T

    if offsets == "Random":
        time_lengths = [x[-1, t_loc] for x in events]
        max_t = np.max(np.array(time_lengths))
        events[:, :, t_loc] -= np.tile(
            np.random.random(events.shape[0]) * max_t, (events.shape[1], 1)
        ).T

    elif offsets is not None:
        events[:, :, t_loc] -= np.tile(np.array(offsets), (events.shape[1], 1)).T

    # Concatenate and sort
    combined_events = np.concatenate(events, axis=0)
    idx = np.argsort(combined_events[:, t_loc])
    combined_events = combined_events[idx]
    num_colisions = 0

    if check_conflicts:
        keep_events = np.ones(len(combined_events))
        for i in range(len(combined_events) - 1):
            ev_1 = combined_events[i]
            ev_1[p_loc] = 0
            ev_2 = combined_events[i + 1]
            ev_2[p_loc] = 0
            if np.array_equal(ev_1, ev_2):
                if np.array_equal(
                    combined_events[i, p_loc], combined_events[i + 1, p_loc]
                ):
                    keep_events[i] = 0
                    i = i + 1
                else:
                    keep_events[i] = 0
                    keep_events[i + 1] = 0
                    i = i + 1
            else:
                keep_events[i] = 1
        num_colisions = len(keep_events) - np.sum(keep_events)
        combined_events = combined_events[keep_events.astype("bool")]

    return combined_events, num_colisions
