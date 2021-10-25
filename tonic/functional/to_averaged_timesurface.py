import numpy as np


def findCell(x, y, bounds):

    # build point
    points = np.array([[x, y]])

    # check for each event if all coordinates are in bounds
    allInBounds = points[:, 0] >= bounds[:, None, 0]
    allInBounds &= points[:, 1] >= bounds[:, None, 1]
    allInBounds &= points[:, 0] < bounds[:, None, 2]
    allInBounds &= points[:, 1] < bounds[:, None, 3]

    # now find out the positions of all nonzero values
    nz = np.nonzero(allInBounds)

    # initialize the result with all nan
    r = np.full(points.shape[0], np.nan)

    # use nz[1] to index event position and nz[0] to tell which cell the event belongs to
    r[nz[1]] = nz[0]
    return int(r)


def to_averaged_timesurface(
    events,
    sensor_size,
    cell_size=10,
    surface_size=3,
    temporal_window=5e5,
    tau=5e3,
    decay="lin",
):
    """Representation that creates averaged timesurfaces for each event for one recording. Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf

    Parameters:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        temporal_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        array of timesurfaces with dimensions (w,h)
    """
    radius = surface_size // 2
    assert surface_size <= cell_size
    assert "x" and "y" and "t" and "p" in events.dtype.names
    n_of_events = len(events)
    n_of_pols = sensor_size[2]

    all_surfaces = np.zeros((n_of_events, n_of_pols, surface_size, surface_size))

    # find how many rows and columns we have in the grid
    rows = -(-sensor_size[0] // cell_size)
    cols = -(-sensor_size[1] // cell_size)
    ncells = rows * cols

    # initialise cell structures
    cells = np.empty(ncells, dtype=object)

    # boundaries for each cell
    xmin = 0
    ymin = 0
    bounds = np.zeros((ncells, 4))
    for i in np.arange(ncells):

        if i != 0 and i % rows == 0:
            xmin = 0
            ymin += cell_size

        bounds[i] = np.array([xmin, ymin, xmin + cell_size, ymin + cell_size])
        xmin += cell_size

    # event loop
    for index, event in enumerate(events):
        x = int(event["x"])
        y = int(event["y"])

        # find the cell
        r = findCell(x, y, bounds)

        # initialise timesurface
        timesurface = np.zeros([surface_size, surface_size])
        timesurface[radius, radius] = 1

        if cells[r]:
            local_memory = np.array(cells[r])

            # find events ej such that tj is in [ti-temporal_window, ti)
            context = local_memory[:, 0] >= event["t"] - temporal_window
            context &= local_memory[:, 0] < event["t"]

            # find events ej such that xj is in [xi-radius,xi+radius]
            context &= local_memory[:, 1] <= x + radius
            context &= local_memory[:, 1] >= x - radius

            # find events ej such that yj is in [yi-radius,yi+radius]
            context &= local_memory[:, 2] <= y + radius
            context &= local_memory[:, 2] >= y - radius

            # taking into consideration different polarities
            context &= local_memory[:, 3] == event["p"]

            # get the neighborhood of center event
            neighborhood = local_memory[context]

            if len(neighborhood) != 0:

                # get unique coordinates
                unique_coord = np.unique(neighborhood[:, 1:3], axis=0)
                for i, coord in enumerate(unique_coord):

                    # get timestamp of matching coordinates from cell
                    match = neighborhood[:, 1] == coord[0]
                    match &= neighborhood[:, 2] == coord[1]

                    # scale x and y to find their position on timesurface
                    scaled_x = int(coord[0] - x + radius)
                    scaled_y = int(coord[1] - y + radius)

                    # for each neighbor use some of decay of past events
                    if decay == "lin":
                        tmp_ts = (neighborhood[match, 0] - event["t"]) / (3 * tau) + 1
                        tmp_ts[tmp_ts < 0] = 0
                        timesurface[scaled_x, scaled_y] = np.sum(tmp_ts)
                    elif decay == "exp":
                        timesurface[scaled_x, scaled_y] = np.sum(
                            np.exp((neighborhood[match, 0] - event["t"]) / tau)
                        )

        else:
            # initialising cell with an empty list
            cells[r] = []

        # save event inside the cell
        cells[r].append((event["t"], x, y, event["p"]))

        # save into all_surfaces
        all_surfaces[index, :, :, :] = timesurface

    return all_surfaces
