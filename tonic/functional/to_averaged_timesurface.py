#! /usr/bin/env python3

import numpy as np
import math

def _gen_pixel_to_cell_mapper(h, w, cell_size):
  """
  Matrix that allows to map a pixel to a cell via indexing.
  """
  matrix = np.zeros((h, w), dtype=np.int32)
  for i in range(h):
    for j in range(w):
      # Since each frame axis is divided in cells of width _K, the cell index related to that axis can be computed by integer division x//_K.
      cell_y, cell_x = i//cell_size, j//cell_size
      # Getting the index associated to cell coordinates.
      matrix[i,j] = cell_y*math.ceil(w/cell_size) + cell_x
  return matrix
  
def _get_cell_mems(events, n_cells, n_pols, pixel_to_cell_mapper):
  """
  Generates the local memories of all cells in the frame.
  """
  cell_mems = [[[] for c in range(n_cells)] for p in range(n_pols)]
  for event in events:
    cell_mems[int(event['p'])][pixel_to_cell_mapper[event['y'], event['x']]].append(event)
  cell_mems = [[np.hstack(cell_mems[p][c]) if cell_mems[p][c] else np.empty((0,)) for c in range(n_cells)] for p in range(n_pols)]
  return cell_mems

def _get_time_surface(event, cell_mem, surface_size=10, temporal_window=500, tau=500, decay='exp'):
  """
  Accumulates the time surfaces of all the events included in the time window of the given event. 
  """
  # Empty cell.
  if cell_mem.size==0:
    return np.zeros((surface_size, surface_size))
  radius = np.array(surface_size//2).astype(cell_mem['x'].dtype)
  # Getting starting timestamp.
  t_start = max(0, event['t'] - temporal_window)
  # Computing surface coordinates for events.
  x_surface, y_surface = (event['x'] + radius - cell_mem['x']), (event['y'] + radius - cell_mem['y'])
  # Filtering events out of surface and time window.
  mask = np.asarray((x_surface>=0) & (x_surface<=2*radius) & (y_surface>=0) & (y_surface<=2*radius) & (cell_mem['t']>=t_start)).nonzero()[0]
  if mask.size==0:
    return np.zeros((surface_size, surface_size))
  # Time surfaces tensor.
  ts = np.zeros((mask.size, surface_size, surface_size), dtype=np.float32)
  # For each event, the corresponding time surface is computed using the coordinates previously calculated. 
  ts[np.arange(mask.size), y_surface[mask], x_surface[mask]] = np.exp(-(event['t'] - cell_mem['t'][mask]).astype(np.float32)/tau) if decay=='exp' else (event['t'] - cell_mem['t'][mask]).astype(np.float32)/(3*tau) + 1
  # The time surfaces are accumulated and returned in output.
  return np.sum(ts, axis=0)

def to_averaged_timesurface(
    events,
    sensor_size=(128, 128, 2),
    cell_size=10,
    surface_size=3,
    temporal_window=5e5,
    tau=5e3,
    decay="exp",
    num_workers=1
):
  """Representation that creates averaged timesurfaces for each event for one recording. Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf
    Parameters:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        temporal_window (int): how far back to look for past events for the time averaging. Expressed in microseconds.
        tau (int): time constant to decay events around occuring event with. Expressed in microseconds.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.
        num_workers (int): number of workers to be deployed on the histograms computation. 
    Returns:
        array of histograms (numpy.Array with shape (n_cells, n_pols, surface_size, surface_size))
  """
  
  assert surface_size <= cell_size
  assert surface_size%2 != 0
  assert "x" and "y" and "t" and "p" in events.dtype.names
  assert decay=='lin' or decay=='exp'

  if num_workers>1:
    try:
      from joblib import Parallel, delayed
    except Exception as e:
      print("Error: num_workers>1 needs joblib installed.")
    use_joblib = True
  else:
    use_joblib = False

  # Getting sensor sizes, number of cells in the frame and initializing the data structures.
  width, height, n_pols = sensor_size
  n_cells = math.ceil(width/cell_size)*math.ceil(height/cell_size)
  histograms = np.zeros((n_cells, n_pols, surface_size, surface_size))
     
  # Matrix for associating an event to a cell via indexing.
  pixel_to_cell_mapper = _gen_pixel_to_cell_mapper(w=width, h=height, cell_size=cell_size)
  
  # Organizing the events in cells (local memories of HATS paper). 
  cell_mems = _get_cell_mems(events=events, n_cells=n_cells, n_pols=n_pols, pixel_to_cell_mapper=pixel_to_cell_mapper)
  
  # Time surfaces associated to each event in a cell.
  get_cell_time_surfaces = lambda cell_mem: np.stack([
    _get_time_surface(event=cell_mem[i], cell_mem=cell_mem[:i], surface_size=surface_size, temporal_window=temporal_window, tau=tau, decay=decay)
    for i in range(cell_mem.shape[0])])

  # Histogram associated to a cell, obtained by accumulatig the time surfaces of each event in the cell.
  get_cell_histogram = lambda cell: np.stack([
    np.sum(get_cell_time_surfaces(cell_mems[pol][cell]), axis=0)/max(sum([cell_mems[p][cell].shape[0] for p in range(n_pols)]), 1)
    if cell_mems[pol][cell].size>0 else
    np.zeros((surface_size, surface_size))
    for pol in range(n_pols)])

  if not use_joblib:
    histograms = np.stack([get_cell_histogram(c) for c in range(n_cells)])
  else:
    histograms = np.stack(
      Parallel(n_jobs=num_workers)(
        delayed(get_cell_histogram)(c) for c in range(n_cells)
      )
    )
  return histograms
