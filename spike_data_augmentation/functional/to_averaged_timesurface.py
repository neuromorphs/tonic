import numpy as np
import time

def findCell(x, y, bounds):
    
    # build point
    points = np.array([[x,y]])
    
    # check for each event if all coordinates are in bounds
    allInBounds = (points[:,0] >= bounds[:,None,0])
    allInBounds &= (points[:,1] >= bounds[:,None,1])
    allInBounds &= (points[:,0] < bounds[:,None,2])
    allInBounds &= (points[:,1] < bounds[:,None,3])

    # now find out the positions of all nonzero values
    nz = np.nonzero(allInBounds)

    # initialize the result with all nan
    r = np.full(points.shape[0], np.nan)
    
    # use nz[1] to index event position and nz[0] to tell which cell the event belongs to
    r[nz[1]] = nz[0]
    return int(r)
        
def to_averaged_timesurface_numpy(
    events,
    sensor_size,
    ordering,
    cell_size=10,
    surface_size=3,
    temporal_window=5e5,
    tau=5e3,
    decay="lin",
    merge_polarities=False,
):
    """Representation that creates averaged timesurfaces for each event for one recording.

    Args:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        time_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        array of timesurfaces with dimensions (w,h)
    """
    radius = surface_size // 2
    assert surface_size <= cell_size
    assert "x" and "y" and "t" and "p" in ordering
#     assert len(sensor_size) == 1
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_of_events = len(events)
    if merge_polarities:
        events[:, p_index] = np.zeros(n_of_events)
    n_of_pols = len(np.unique(events[:, p_index]))
 
    # initialising matrix containing all averaged time surfaces
    all_surfaces = np.zeros(
        (n_of_events, n_of_pols, surface_size, surface_size)
    )
    
    # find how many rows and columns we have in the grid
    rows = -(-sensor_size[0] // cell_size)
    cols = -(-sensor_size[1] // cell_size)
    ncells = rows * cols
    
    # boundaries for each cell
    bounds = np.zeros((ncells,4))
    xmin = 0;
    ymin = 0;
    for i in np.arange(ncells):
        
        if i != 0 and i % rows == 0:
            xmin=0
            ymin += cell_size
            
        bounds[i] = np.array([xmin,ymin,xmin+cell_size,ymin+cell_size])
        xmin += cell_size
    
    # initialise cell structures
    cells = np.empty(ncells, dtype=object)
    for index, event in enumerate(events):
        x = int(event[x_index])
        y = int(event[y_index])
        t = int(event[t_index])
        p = int(event[p_index])
        
        # find the cell
        r = findCell(x, y, bounds)
        
        if cells[r]:
            local_memory = np.array(cells[r]).T
        
            # find events ej such that tj is in [ti-temporal_window, ti)
            context  t= (local_memory[0] < t)
            context &= (local_memory[0] >= t-temporal_window)
            
            # find events ej such that xj is in [xi-radius,xi+radius]
            context &= (local_memory[1] <= x+radius)
            context &= (local_memory[1] >= x-radius) 
            
            # find events ej such that yj is in [yi-radius,yi+radius]
            context &= (local_memory[2] <= y+radius)
            context &= (local_memory[2] >= y-radius)
            
            # taking into consideration different polarities
            context &= (local_memory[3] == p)
            
            # get non-zero values that denote the relevant events inside neighborhood
            relevant_events = np.nonzero(context)
            
            # local_memory[relevant_events] contains the neighborhood of events needed to do the time surface
            timestamp_context = local_memory[0][relevant_events] - t 
            
            # step missing here that "averages" the event times to be able to shape the timesurface: they do a sum over all the events according to HATS paper
            # figure suggests but I find it odd to use a sum of decays
            
            # apply decay on each value in the relevant context
            if decay == "lin":
                timestamp_context /= (3 * tau) + 1
                timestamp_context[timestamp_context < 0] = 0
            elif decay == "exp":
                timestamp_context = np.exp(timestamp_context / tau)
                timestamp_context[timestamp_context < (-3 * tau)] = 0
        else:
            # initialising cell with an empty list
            cells[r] = []
            
        # save event inside the cell
        cells[r].append((t,x,y,p))
        
        # save into all_surfaces
#         all_surfaces[index, :, :, :] = timesurface
        
    return all_surfaces
