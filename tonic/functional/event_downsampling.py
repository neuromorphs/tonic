import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

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

    events["x"] = events["x"] * spatial_factor[1]
    events["y"] = events["y"] * spatial_factor[0]

    return events

def time_bin_numpy(events: np.ndarray, time_bin_interval: float):
    """Temporally downsample the events into discrete time bins as stipulated by time_bin_intervals.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        time_bin_interval (float): time bin size for events e.g. every 0.5 ms: time_bin_interval = 0.5.
    Returns:
        the input events with rewritten timestamps.
    """
    
    events = events.copy()
    reciprocal_interval = 1 / time_bin_interval
    events["t"] = np.round(events["t"] * reciprocal_interval, 0) / reciprocal_interval
    
    return events

def feedback_inhibition(events: np.ndarray, target_size: tuple, dt: float, buffer_layer_parameters: dict, inhibitory_layer_parameters: dict,
                        buffer_postsynaptic_strength: float = 0.005, inhibitory_postsynaptic_strength: float = 4.0, delay: int = 1):
    """Feedback inhibition to eliminate high-frequency noise inspired by locust brains.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): step size for simulation, in ms.
        buffer_layer (dict): dictionary of neuron population parameters.
        inhibitory_layer (dict): dictionary of neuron population parameters.
        buffer_postsynaptic_strength (float): synaptic strength of buffer to inhibitory neuron populations.
        inhibitory_postsynaptic_strength (float): synaptic strength of inhibitory to buffer neuron populations.
        delay (int): dendritic delay in timesteps.
    Returns:
        the downsampled events after feedback inhibition.
        
    * Define leaky integrate and fire (LIF) and linear threshold (LT) neuron models
    V(t+1) = p*V(t) + I_syn
    V (ndarray): membrane potential of shape target_size, initial value of V set at zero
    p (float): persistence variable or leak factor is a float that is an element of {0, 1}
    I_syn (nd_array): injection current to layer of neurons of shape target_size
    
    where
    I_syn = g_exc * w_e * a_e(t-delta_e) - g_inh * w_i * a_i(t-delta_i)
    g_{exc, inh} (float): synaptic strength that is an element of {0, 1}
    a_{e, i} (ndarray): excitatory neuron population activity
    delta_{e, i} (int): timestep delays
    
    * Activity updates
    LT model -> a=v(t+1) if v(t+1)>=theta, else a=0
    LIF model -> a=1 if v(t+1)>=theta, else a=0
    After hyperpolarisation, v'(t+1) = v(t+1) - alpha where alpha is the amplitude of hyperpolarisation
    
    Assumptions: One-to-one synaptic connectivities
                 Homogeneous layer-wise neuron and synapse population attributes
    """
    
    assert "x" and "y" in events.dtype.names
    
    events = events.copy()
    
    events = time_bin_numpy(events, dt)
    
    # Parameter definitions
    assert "p" and "theta" and "alpha" in buffer_layer_parameters.keys()
    assert "p" in inhibitory_layer_parameters.keys()
    
    # All event times
    event_times = np.unique(events["t"])
    
    # Separate by polarity
    events_positive = events[events["p"] == 1]
    events_negative = events[events["p"] == 0]
    
    assert isinstance(delay, int)
    
    simulation_time = event_times[-1] + (2 + delay) * dt + dt
    simulation_timesteps = int(np.ceil(simulation_time / dt))
    
    # Ravelled array of neuron population
    buffer_layer = np.zeros((simulation_timesteps, np.prod(target_size), 2))
    inhibitory_layer = np.zeros((simulation_timesteps, np.prod(target_size), 2))
    
    for t, time in enumerate(np.arange(0, event_times[-1] + dt, dt)):
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        # Ravelled coordinates
        coordinate_pos = xy_pos["y"] * target_size[0] + xy_pos["x"]
        coordinate_neg = xy_neg["y"] * target_size[0] + xy_neg["x"]
        
        # Spike potential because of event causes activity in buffer neuron population (LIF neuron model)
        buffer_layer[t, coordinate_pos, 0] += 1
        buffer_layer[t, coordinate_neg, 1] += 1
        
        # Buffer layer activity updates
        buffer_layer[t+1,:,:] = buffer_layer[t,:,:] * buffer_layer_parameters["p"]
        
        buffer_layer[t+1,:,:] = np.where(buffer_layer[t+1,:,:] >= buffer_layer_parameters["theta"], 
                                         buffer_layer[t+1,:,:] - buffer_layer_parameters["alpha"], buffer_layer[t+1,:,:])
        
        # Current in B-I synapse
        buffer_to_inhibitory_synapse = (buffer_layer[t,:,:] * buffer_postsynaptic_strength).clip(min=0)
        
        # Inhibitory layer activity updates (LT neuron model)
        inhibitory_layer[t+1,:,:] = inhibitory_layer[t,:,:] * inhibitory_layer_parameters["p"] + buffer_to_inhibitory_synapse
        
        # Current in I-B synapse
        inhibitory_to_buffer_synapse = (inhibitory_layer[t+1,:,:] * inhibitory_postsynaptic_strength).clip(min=0)

        # Feedback into buffer layer
        buffer_layer[t+2+delay] -= inhibitory_to_buffer_synapse
        
    # First instance of spike in buffer layer for both polarities
    first_instance = [np.nonzero(buffer_layer[:,:,p])[0][0] for p in range(2)]
    
    # New event times and indices
    spike_timestep_pos, index_pos = np.where(np.abs(buffer_layer[first_instance[0]+1:,:,0]) >= buffer_layer_parameters["theta"])
    spike_timestep_neg, index_neg = np.where(np.abs(buffer_layer[first_instance[1]+1:,:,1]) >= buffer_layer_parameters["theta"])
    
    # Restructuring from numpy array to structured array
    y_pos, x_pos = np.unravel_index(index_pos, np.flip(target_size))
    y_neg, x_neg = np.unravel_index(index_neg, np.flip(target_size))
    
    spike_timestep_pos = (spike_timestep_pos + first_instance[0] + 1) * dt
    spike_timestep_neg = (spike_timestep_neg + first_instance[1] + 1) * dt
    
    events_pos = np.column_stack((x_pos, y_pos, np.ones((len(spike_timestep_pos),1)), spike_timestep_pos.T))
    events_neg = np.column_stack((x_neg, y_neg, np.zeros((len(spike_timestep_neg),1)), spike_timestep_neg.T))
    
    events_new = np.row_stack((events_pos, events_neg))
    events_new = events_new[events_new[:,-1].argsort()]
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)

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
    
    # Create time bins according to differentiator
    events = time_bin_numpy(events, dt / differentiator_time_bins)
    
    # Call integrator method
    events_integrated = integrator_downsample(events=events, sensor_size=sensor_size, target_size=target_size, 
                                              noise_threshold=noise_threshold)
    
    # All event times
    events_adjusted = time_bin_numpy(events_integrated, dt)
    
    event_times = np.unique(events_adjusted["t"])
    
    # Separate by polarity
    events_positive = events_adjusted[events_adjusted["p"] == 1]
    events_negative = events_adjusted[events_adjusted["p"] == 0]
    
    frame_histogram = np.zeros((len(event_times), *np.flip(target_size), 2))
    
    for t, time in enumerate(event_times):
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        frame_histogram[t,:,:,1] += np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
        frame_histogram[t,:,:,0] += np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
        
    # Differences between subsequent frames
    frame_differences = np.diff(frame_histogram, axis=0).clip(min=0)

    # Restructuring numpy array to structured array
    time_index, y_new, x_new, polarity_new = np.nonzero(frame_differences)
    
    events_new = np.column_stack((x_new, y_new, polarity_new, event_times[time_index]))
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)
    
def integrator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, noise_threshold: int = 0):
    """Downsample using an integrate-and-fire (I-F) neuron model with a noise threshold similar to 
    the membrane potential threshold in the I-F model. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        noise_threshold (int): number of events before a spike representing a new event is emitted.
    Returns:
        the downsampled input events using the differentiator method.
    """
        
    assert "x" and "y" and "t" in events.dtype.names
    
    events = events.copy()
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]
    
    # All event times
    event_times = np.unique(events["t"])
    
    # Separate by polarity
    events_positive = events[events["p"] == 1]
    events_negative = events[events["p"] == 0]
    
    # Running buffer of events in each pixel
    frame_spike = np.zeros(np.flip(target_size))
    
    events_new = []
    
    for time in event_times:
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        # Sum in 2D space using histogram
        frame_histogram = np.subtract(np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0],
                                      np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0])
        
        frame_spike += frame_histogram
        
        coordinates_pos = np.stack(np.nonzero(np.maximum(frame_spike >= noise_threshold, 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-frame_spike >= noise_threshold, 0))).T
        
        # Reset spiking coordinates to zero
        frame_spike[coordinates_pos] = 0
        frame_spike[coordinates_neg] = 0
        
        # Add to event buffer
        events_new.append(np.column_stack((coordinates_pos, np.ones((coordinates_pos.shape[0],1)), time*np.ones((coordinates_pos.shape[0],1)))))
        events_new.append(np.column_stack((coordinates_neg, np.zeros((coordinates_neg.shape[0],1)), time*np.ones((coordinates_neg.shape[0],1)))))
        
    events_new = np.concatenate(events_new.copy())
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)
