import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from tonic.functional.to_frame import to_frame_numpy

def differentiator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, 
                              differentiator_time_bins: int = 2, noise_threshold: int = 0):
    """Spatio-temporally downsample using the integrator method coupled with a differentiator to effectively 
    downsample large object sizes relative to downsampled pixel resolution in the DVS camera's visual field.
    
    Incorporates the paper Ghosh et al. 2023, Insect-inspired Spatio-temporal Downsampling of Event-based Input,
    https://doi.org/10.1145/3589737.3605994
    
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
        the spatio-temporally downsampled input events using the differentiator method.
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
        event_hist_pos = (np.maximum(event_histogram >= noise_threshold, 0)).clip(max=1)
        event_hist_neg = (-np.minimum(-event_histogram >= noise_threshold, 0)).clip(max=1)
        
        frame_histogram[time,...,1] += event_hist_pos
        frame_histogram[time,...,0] += event_hist_neg
        
    # Differences between subsequent frames
    frame_differences = (np.diff(frame_histogram, axis=0)).clip(min=0)
    
    # Restructuring numpy array to structured array
    time_index, y_new, x_new, polarity_new = np.nonzero(frame_differences)
    
    events_new = np.column_stack((x_new, y_new, polarity_new.astype(dtype=bool), time_index * dt))
    
    names = ["x", "y", "p", "t"]
    formats = ['i4', 'i4', 'i4', 'i4']
    
    dtype = np.dtype({'names': names, 'formats': formats})
    
    return unstructured_to_structured(events_new.copy(), dtype=dtype)
    
def integrator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, noise_threshold: int = 0, 
                          differentiator_call: bool = False):
    """Spatio-temporally downsample using with the following steps:
    
    1. Differencing of ON and OFF events to counter camera shake or jerk.
    2. Use an integrate-and-fire (I-F) neuron model with a noise threshold similar to 
    the membrane potential threshold in the I-F model to eliminate high-frequency noise.
    
    Multiply x/y values by a spatial_factor obtained by dividing sensor size by the target size.
    
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
        the spatio-temporally downsampled input events using the integrator method.
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
    
    # Compute all histograms at once
    all_frame_histograms = to_frame_numpy(events, sensor_size=(*target_size, 2), time_window=dt)
    
    # Subtract the channels for ON/OFF differencing
    frame_histogram_diffs = all_frame_histograms[:, 1] - all_frame_histograms[:, 0]
    
    frame_spike = np.zeros(np.flip(target_size))
    event_histogram = []
    
    events_new = []
    
    for time, frame_histogram in enumerate(frame_histogram_diffs):
    
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
        
        names = ["x", "y", "p", "t"]
        formats = ['i4', 'i4', 'i4', 'i4']
        
        dtype = np.dtype({'names': names, 'formats': formats})
        
        return unstructured_to_structured(events_new.copy(), dtype=dtype)