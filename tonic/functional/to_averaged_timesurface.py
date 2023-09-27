#! /usr/bin/env python3

from math import ceil

import numpy as np


def _get_ts(event, locmem, time_window, tau, surface_size, decay):
    rho = surface_size // 2
    t_i, t_j = event["t"].astype(np.float32), locmem["t"].astype(np.float32)
    # Starting time stamp, calculated subtracting the time window from the event timestamp.
    t_start = max(0, t_i - time_window)
    # Relative coordinates in the time surfaces.
    ts_x, ts_y = locmem["x"] - event["x"], locmem["y"] - event["y"]
    # Including only the events in the neighbourhood and in the time window.
    mask = np.asarray(
        (np.abs(ts_x) <= rho) & (np.abs(ts_y) <= rho) & (t_j >= t_start)
    ).nonzero()[0]
    # For each event in the local memory that belongs to the spatial and temporal windows and for the current event, a time surface is generated.
    locmem_ts = np.zeros((1 + len(mask), surface_size, surface_size), dtype=np.float32)
    if len(mask) > 0:
        locmem_ts[np.arange(len(mask)), ts_y[mask] + rho, ts_x[mask] + rho] = (
            np.exp(-(t_i - t_j[mask]) / tau)
            if decay == "exp"
            else -(t_i - t_j[mask]) / (3 * tau) + 1
        )
    # Adding the current event time surface.
    locmem_ts[-1, rho, rho] += 1
    # The accumulated time surfaces are returned.
    return np.sum(locmem_ts, axis=0)


def _map_to_locmems(events, sensor_size, cell_size):
    w, h, npols = sensor_size
    wgrid, hgrid = ceil(w / cell_size), ceil(h / cell_size)
    ncells = int(wgrid * hgrid)
    px_to_cell = lambda y, x: int((y // cell_size) * wgrid + x // cell_size)
    # Mapping events to local memories.
    locmems = [[[] for p in range(npols)] for c in range(ncells)]
    for event in events:
        locmems[px_to_cell(int(event["y"]), int(event["x"]))][
            max(0, int(event["p"]))
        ].append(event)
    # Converting the lists in structured NumPy arrays.
    locmems = [
        [
            np.stack(locmems[c][p]) if locmems[c][p] else np.empty((0,))
            for p in range(npols)
        ]
        for c in range(ncells)
    ]
    return locmems


def to_averaged_timesurface_numpy(
    events,
    sensor_size,
    cell_size,
    surface_size,
    time_window,
    tau,
    decay,
):
    """Representation that creates averaged timesurfaces for each event for one recording.

    Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf
    Parameters:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        time_window (int): how far back to look for past events for the time averaging. Expressed in microseconds.
        tau (int): time constant to decay events around occuring event with. Expressed in microseconds.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    Returns:
        array of histograms (numpy.Array with shape (n_cells, n_pols, surface_size, surface_size))
    """

    assert surface_size <= cell_size
    assert surface_size % 2 != 0
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert decay == "lin" or decay == "exp"

    # Organizing the events in cells which are, then, saved as NumPy arrays.
    locmems = _map_to_locmems(events, sensor_size, cell_size)
    w, h, npols = sensor_size
    ncells = int(ceil(w / cell_size) * ceil(h / cell_size))
    hist = np.zeros((ncells, npols, surface_size, surface_size), dtype=np.float32)
    # Now we have fun: we cycle on the local memories and generate the histogram corresponding to each of them.
    for c in range(ncells):
        for p in range(npols):
            hist[c, p, :, :] = np.sum(
                np.stack(
                    [
                        _get_ts(
                            locmems[c][p][i],
                            locmems[c][p][:i],
                            time_window,
                            tau,
                            surface_size,
                            decay,
                        )
                        for i in range(len(locmems[c][p]))
                    ]
                    if locmems[c][p].size > 0
                    else np.zeros((surface_size, surface_size), dtype=np.float32)
                ),
                axis=0,
            ) / max(1, locmems[c][p].size)
    return hist
