import numpy as np
from joblib import Parallel, delayed, cpu_count
import math

def _gen_pixel_to_cell_mapper(w, h, K, px_dtype):
  matrix = np.zeros((h, w), dtype=px_dtype)
  for i in range(h):
    for j in range(w):
      # Since each frame axis is divided in cells of width _K, the cell index related to that axis can be computed by integer division x//_K.
      cell_y, cell_x = i//K, j//K
      # Getting the index associated to cell coordinates.
      matrix[i,j] = cell_y*math.ceil(w/K) + cell_x
  return matrix
  
def _get_cell_mems(events, n_cells, n_pols, pixel_to_cell_mapper):
  cell_mems = [[[] for c in range(n_cells)] for p in range(n_pols)]
  for event in events:
    cell_mems[max(event['p'], 0)][pixel_to_cell_mapper[event['y'], event['x']]].append(event)
  return cell_mems[:]

def _bsearch_window(t_start, cell_mem):
  l, r, start = 0, len(cell_mem)-1, 0
  while l<=r:
    m = (l+r)//2
    if cell_mem[m]['t'] > t_start:
      r, start = m-1, m
    elif cell_mem[m]['t'] < t_start:
      l = m+1
    else:
      start = m
      break
  return start

def _get_time_surface(event, cell_mem, radius, temporal_window, tau):
  ts = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
  # Getting starting timestamp.
  t_start = max(0, event['t'] - temporal_window)
  check_coords = lambda x, y: x>=0 and y>=0 and x <= 2*radius and y <= 2*radius
  # Getting start timestamp of temporal window.
  start = _bsearch_window(t_start=t_start, cell_mem=cell_mem)
  for mem_event in cell_mem[start:]:
    # Getting event coordinates in the neighbourhood.
    x, y = (event['x'] - mem_event['x']) + radius, (event['y'] - mem_event['y']) + radius
    # Check if event is in neighourbhood and, if so, adding it to the time surface.
    if check_coords(x, y):
      ts[y,x] += np.exp(-(event['t'] - mem_event['t']).astype(np.float32)/tau)
  return ts
  
def to_averaged_timesurface(
    events,
    sensor_size,
    cell_size=10,
    surface_size=3,
    temporal_window=5e5,
    tau=5e3,
    decay="lin"
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
        array of histograms (numpy.Array with shape (n_cells, n_pols, surface_size, surface_size))
  """
  
  radius = surface_size // 2
  assert surface_size > 0 and surface_size <= cell_size
  assert "x" and "y" and "t" and "p" in events.dtype.names
  assert decay=='lin' or decay=='exp'

  # Temporary division of resources.
  ts_jobs, cells_jobs = round(cpu_count()*0.2), round(cpu_count()*0.6)
  
  # Getting sensor sizes, number of cells in the frame and initializing the data structures.
  width, height, n_pols = sensor_size
  n_cells = math.ceil(width/surface_size)*math.ceil(height/surface_size)
  histograms = np.zeros((n_cells, n_pols, surface_size, surface_size))
     
  # Matrix for associating an event to a cell via indexing.
  pixel_to_cell_mapper = _gen_pixel_to_cell_mapper(w=width, h=height, K=surface_size, px_dtype=events["x"].dtype)
  
  # Organizing the events in cells (local memories of HATS paper). 
  cell_mems = _get_cell_mems(events=events, n_cells=n_cells, n_pols=n_pols, pixel_to_cell_mapper=pixel_to_cell_mapper)

  use_joblib = True
  
  # Time surfaces associated to each event in a cell.
  if not use_joblib:
    get_cell_time_surfaces = lambda cell_mem: np.stack([
      _get_time_surface(event=cell_mem[i], cell_mem=cell_mem[:i], radius=radius, temporal_window=temporal_window, tau=tau)
      for i in range(len(cell_mem))])
  else:  
    get_cell_time_surfaces = lambda cell_buffer: np.stack(
      Parallel(n_jobs=ts_jobs)(
        delayed(_get_time_surface)(cell_buffer[i], cell_buffer[:i], radius, temporal_window, tau)
        for i in range(len(cell_buffer))
      )
    )

  # Histogram associated to a cell, obtained by accumulatig the time surfaces of each event in the cell.
  get_cell_histogram = lambda cell: np.stack([
    np.sum(get_cell_time_surfaces(cell_mems[pol][cell]), axis=0)/max(sum([len(cell_mems[pol][cell]) for pol in range(n_pols)]), 1)
    if cell_mems[pol][cell] else
    np.zeros((2*radius+1, 2*radius+1))
    for pol in range(n_pols)])

  if not use_joblib:
    histograms = np.stack([get_cell_histogram(c) for c in range(n_cells)])
  else:
    histograms = np.stack(
      Parallel(n_jobs=cells_jobs)(
        delayed(get_cell_histogram)(c) for c in range(n_cells)
      )
    )
  
  return histograms  

if __name__ == "__main__":
  import tonic
  from timeit import default_timer
  ds = tonic.datasets.NMNIST(save_to="./train_data", train=True)
  events, labels = ds[0]
  start = default_timer()
  h = to_averaged_timesurface(events, ds.sensor_size, cell_size=5, surface_size=5) 
  print(f"histogram shape: {h.shape}, time: {(default_timer()-start)*1e3:.2f}ms.")
  
