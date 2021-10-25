import numpy as np


# from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/preprocess.py#L188
def identify_hot_pixel(events: np.ndarray, hot_pixel_frequency: float):
    """Identifies pixels that fire above above a certain frequency, averaged across
    whole event recording. Such _hot_ pixels are sometimes caused by faulty hardware.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        hot_pixel_frequency: number of spikes per pixel allowed for the recording, any pixel
                             firing above that number will be deactivated.

    Returns:
        list of (x/y) coordinates for excessively firing pixels.
    """

    assert "x" and "y" and "t" in events.dtype.names

    total_time = events["t"][-1] - events["t"][0]

    hist = np.histogram2d(
        events["x"],
        events["y"],
        bins=(np.arange(events["y"].max() + 1), np.arange(events["x"].max() + 1)),
    )[0]
    max_occur = hot_pixel_frequency * total_time * 1e-6
    hot_pixels = np.asarray((hist > max_occur).nonzero()).T

    return hot_pixels


def drop_pixel_numpy(events: np.ndarray, coordinates):
    """Drops events for pixel locations that fire

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events. This function requires 'x' and
                  'y' to be in the ordering
        coordinates: list of (x,y) coordinates for which all events will be deleted.

    Returns:
        subset of original events.
    """

    assert "x" and "y" in events.dtype.names

    dropped_pixel_mask = np.full((events.shape[0]), False, dtype=bool)
    for x, y in coordinates:
        current_mask = np.logical_and(events["x"] == x, events["y"] == y)
        dropped_pixel_mask = np.logical_or(current_mask, dropped_pixel_mask)

    return events[np.invert(dropped_pixel_mask)]
