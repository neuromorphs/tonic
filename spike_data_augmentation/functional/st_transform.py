import numpy as np

from .utils import guess_event_ordering_numpy

def st_transform(
    events,
    ordering,
    spatial_transform,
    temporal_transform,
    Roll = False,
    sensor_size=(346, 260)):
    """
    Transform all events spatial and temporal locations based on
    given spatial transform matrix and temporal transform vector.

    Arguments:

    events - ndarray of shape [num_events, num_event_channels]
    ordering - ordering of the event tuple inside of events, if None
    the system will take a guess through
    guess_event_ordering_numpy. This function requires 't', 'x', 'y'
    to be in the ordering
    spatial_transform - 3x3 matrix which can be used to perform rigid (translation and rotation),
    non-rigid (scaling and shearing), and non-affine transformations. Generic to user input.
    temporal_transform - scale time between events and offset temporal location based on 2 member vector.
    Used as arguments to time_skew method.
    Roll - boolean input to determine if transformed events will be translated across sensor boundaries (True).
    Otherwise, events will be clipped at sensor boundaries.
    sensor_size - tuple which stipulates sensor size and used to determine tranform limits.
    Returns:
    events - returns the input events with tranformed temporal and spatial location
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
        assert "x" and "y" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")

    number_events = events.shape[0]

    ones_vec = np.ones((number_events,1))
    events_homog_coord = np.hstack((events[:,x_index].reshape(number_events,1),events[:,y_index].reshape(number_events,1),ones_vec))
    #spatial transform
    events_spatial_transform = np.matmul(spatial_transform,events_homog_coord.T)
    #Roll coordinates if specified
    events_spatial_transform_X = events_spatial_transform[x_index,:]
    events_spatial_transform_Y = events_spatial_transform[y_index,:]

    outOfRange_eventsX_P = np.where(events_spatial_transform[x_index,:] >= sensor_size[1]) # Out Of Range coordinates based on imDim
    outOfRange_eventsX_N = np.where(events_spatial_transform[x_index,:] < 0)
    outOfRange_eventsY_P = np.where(events_spatial_transform[y_index,:] >= sensor_size[0])
    outOfRange_eventsY_N = np.where(events_spatial_transform[y_index,:] < 0)

    if Roll:
        events_spatial_transform_X[outOfRange_eventsX_P] -= sensor_size[0] # Roll X right
        events_spatial_transform_X[outOfRange_eventsX_N] += sensor_size[0] # Roll X left
        events_spatial_transform_Y[outOfRange_eventsY_P] -= sensor_size[1] # Roll Y down
        events_spatial_transform_Y[outOfRange_eventsY_N] += sensor_size[1] # Roll Y up
    else:
        events_spatial_transform_X[outOfRange_eventsX_P] = sensor_size[0] # Clip X pos.
        events_spatial_transform_X[outOfRange_eventsX_N] = 0 # Clip X neg.
        events_spatial_transform_Y[outOfRange_eventsY_P] = sensor_size[1] # Clip Y pos.
        events_spatial_transform_Y[outOfRange_eventsY_N] = 0 # Clip Y neg.

    events[:,x_index] = events_spatial_transform_X.T
    events[:,y_index] = events_spatial_transform_Y.T

    events = time_skew(events, ordering, temporal_transform[0], temporal_transform[1])

    return events
