import numpy as np


def identify_hot_pixel(events: np.ndarray, sensor_size, ordering: str, hot_pixel_frequency: float):
    """Identifies pixels that fire above above a certain frequency, averaged across 
    whole event recording. Such _hot_ pixels are sometimes caused by faulty hardware.
    
    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events. This function requires 'x' and
                  'y' to be in the ordering
        hot_pixel_frequency: number of spikes per pixel allowed for the recording, any pixel
                             firing above that number will be deactivated. 
        
    Returns:
        list of (x/y) coordinates for excessively firing pixels.
    """

    assert "x" and "y" and "t" in ordering
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    
    total_time = events[:,t_index][-1] - events[:,t_index][0]
    
    hist = np.histogram2d(events[:,x_index], events[:,y_index], bins=(np.arange(sensor_size[1]+1), np.arange(sensor_size[0]+1)))[0]
    max_occur = hot_pixel_frequency * total_time * 1e-6
    hot_pixels = np.asarray((hist > max_occur).nonzero()).T
    return hot_pixels


# from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/preprocess.py#L188
def drop_pixel_numpy(events: np.ndarray, ordering: str, coordinates):
    """Drops events for pixel locations that fire 

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        ordering: ordering of the event tuple inside of events. This function requires 'x' and
                  'y' to be in the ordering
        coordinates: list of (x,y) coordinates for which all events will be deleted.

    Returns:
        subset of original events.
    """

    assert "x" and "y" in ordering
    x_index = ordering.find("x")
    y_index = ordering.find("y")
            
    dropped_pixel_mask = np.full((events.shape[0]), False, dtype=bool)
    for x, y in coordinates:
        xs = events[:, x_index] == x
        ys = events[:, y_index] == y
        events = events[~(xs & ys)]
    return events
#         current_mask = np.logical_and(events[:, x_index] == x, events[:, y_index] == y)
#         dropped_pixel_mask = np.logical_or(current_mask, dropped_pixel_mask)

    return events[np.invert(dropped_pixel_mask),:]
