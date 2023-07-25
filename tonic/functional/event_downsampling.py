import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from tonic.slicers import slice_events_by_time

def naive_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple):
    """Downsample the classic "naive" Tonic way. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
                             
    Returns:
        the downsampled input events.
    """
    
    assert "x" and "y" in events.dtype.names
    
    events = events.copy()
    
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]

    return events

def differentiator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, 
                              differentiator_time_bins: int = 2, noise_threshold: int = 0):
    """Downsample using an integrate-and-fire (I-F) neuron model with an additional differentiator 
    with a noise threshold similar to the membrane potential threshold in the I-F model. Multiply 
    x/y values by a spatial_factor obtained by dividing sensor size by the target size.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): step size for simulation, in ms.
        differentiator_time_bins (int): number of equally spaced time bins with respect to the dt 
                                        to be used for the differentiator.
        noise_threshold (int): number of events before a spike representing a new event is emitted.
        
    Returns:
        the downsampled input events using the differentiator method.
    """
        
    assert "x" and "y" and "t" in events.dtype.names
    assert np.logical_and(np.remainder(differentiator_time_bins, 1) == 0, differentiator_time_bins >= 1)
    
    events = events.copy()
    
    # Call integrator method
    dt_scaling, events_integrated = integrator_downsample(events, sensor_size=sensor_size, target_size=target_size, 
                                                          dt=(dt / differentiator_time_bins), 
                                                          noise_threshold=noise_threshold, differentiator_call=True)
    
    if dt_scaling:
        dt *= 1000
        
    num_frames = int(events_integrated[-1][0] // dt + 1)
    frame_histogram = np.zeros((num_frames, *np.flip(target_size), 2))
        
    for event in events_integrated:
        differentiated_time, event_histogram = event
        time = int(differentiated_time // dt)
        
        # Separate events based on polarity and apply Heaviside
        event_hist_pos = (np.maximum(event_histogram, 0)).clip(max=1)
        event_hist_neg = (-np.minimum(event_histogram, 0)).clip(max=1)
        
        frame_histogram[time,...,1] += event_hist_pos
        frame_histogram[time,...,0] += event_hist_neg
        
    # Differences between subsequent frames
    frame_differences = np.diff(frame_histogram, axis=0).clip(min=0)
    
    # Restructuring numpy array to structured array
    time_index, y_new, x_new, polarity_new = np.nonzero(frame_differences)
    
    events_new = np.column_stack((x_new, y_new, polarity_new.astype(dtype=bool), time_index * dt))
    
    return unstructured_to_structured(events_new.copy(), dtype=[("x", "<i4"), ("y", "<i4"), ("p", "<i4"), ("t", "<i4")])
    
def integrator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, noise_threshold: int = 0, 
                          differentiator_call: bool = False):
    """Downsample using an integrate-and-fire (I-F) neuron model with a noise threshold similar to 
    the membrane potential threshold in the I-F model. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): temporal resolution of events in milliseconds.
        noise_threshold (int): number of events before a spike representing a new event is emitted.
        differentiator_call (bool): Preserve frame spikes for differentiator method in order to optimise 
                                    differentiator method.
        
    Returns:
        the downsampled input events using the integrator method.
    """
    
    assert "x" and "y" and "t" in events.dtype.names
    assert isinstance(noise_threshold, int)
    assert dt is not None
    
    events = events.copy()
    
    if np.issubdtype(events["t"].dtype, np.integer):
        dt *= 1000
        dt_scaling = True
    
    if differentiator_call:
        assert dt // events["t"][-1] == 0
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]
    
    # Re-format event times to new temporal resolution
    events_sliced = slice_events_by_time(events, time_window=dt)
    
    # Running buffer of events in each pixel
    frame_spike = np.zeros(np.flip(target_size))
    event_histogram = []
    
    events_new = []
    
    for time, event in enumerate(events_sliced):
        # Separate by polarity
        xy_pos = event[event["p"] == 1]
        xy_neg = event[event["p"] == 0]
        
        # Sum in 2D space using histogram
        frame_histogram = np.subtract(np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0],
                                      np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0])
        
        frame_spike += frame_histogram
            
        coordinates_pos = np.stack(np.nonzero(np.maximum(frame_spike >= noise_threshold, 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-frame_spike >= noise_threshold, 0))).T
        
        if np.logical_or(coordinates_pos.size, coordinates_neg.size).sum():
        
            # For optimising differentiator
            event_histogram.append((time*dt, frame_spike.copy()))
            
            # Reset spiking coordinates to zero
            frame_spike[coordinates_pos[:,0], coordinates_pos[:,1]] = 0
            frame_spike[coordinates_neg[:,0], coordinates_neg[:,1]] = 0
            
            # Restructure events
            events_new.append(np.column_stack((np.flip(coordinates_pos, axis=1), np.ones((coordinates_pos.shape[0],1)).astype(dtype=bool), 
                                                (time*dt)*np.ones((coordinates_pos.shape[0],1)))))
            
            events_new.append(np.column_stack((np.flip(coordinates_neg, axis=1), np.zeros((coordinates_neg.shape[0],1)).astype(dtype=bool), 
                                                (time*dt)*np.ones((coordinates_neg.shape[0],1)))))
        
    if differentiator_call:
        return dt_scaling, event_histogram
    else:
        events_new = np.concatenate(events_new.copy())
        return unstructured_to_structured(events_new.copy(), dtype=[("x", "<i4"), ("y", "<i4"), ("p", "<i4"), ("t", "<i4")])