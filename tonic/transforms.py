from dataclasses import dataclass
from . import functional
from typing import Callable, Optional, Tuple
import numpy as np


class Compose:
    """Composes several transforms together.

    Parameters:
        transforms (list of ``Transform`` objects): list of transform(s) to compose. Even when using a single transform, the Compose wrapper is necessary.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Denoise(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms: Callable):
        self.transforms = transforms

    def __call__(self, events):
        for t in self.transforms:
            events = t(events)
        return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@dataclass
class RandomCrop:
    """Crops the sensor size to a smaller sensor in a random location.

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Parameters:
        target_size: size of the sensor that was used [W',H']
    """

    target_size: Tuple[int, int]
    sensor_size: Tuple[int, int]
    ordering: str

    def __call__(self, events):
        return functional.crop_numpy(
            events=events,
            sensor_size=self.sensor_size,
            ordering=self.ordering,
            target_size=self.target_size,
        )


@dataclass
class Denoise:
    """Drops events that are 'not sufficiently connected to other events in the recording.'
    In practise that means that an event is dropped if no other event occured within a spatial neighbourhood
    of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to filter noisy recordings
    where events occur isolated in time.

    Parameters:
        filter_time (float): maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.
    """

    ordering: str
    filter_time: float = 10000

    def __call__(self, events):
        return functional.denoise_numpy(
            events=events, ordering=self.ordering, filter_time=self.filter_time,
        )


@dataclass
class DropEvent:
    """Randomly drops events with drop_probability.

    Parameters:
        drop_probability (float): probability of dropping out event.
        random_drop_probability (bool): randomize the dropout probability
                                 between 0 and drop_probability.
    """

    drop_probability: float = 0.5
    random_drop_probability: bool = False

    def __call__(self, events):
        return functional.drop_event_numpy(
            events, self.drop_probability, self.random_drop_probability
        )


@dataclass
class DropPixel:
    """Drops events for individual pixels. If the locations of pixels to be dropped is known, a
    list of x/y coordinates can be passed directly. Alternatively, a cutoff frequency for each pixel can be defined
    above which pixels will be deactivated completely. This prevents so-called _hot pixels_ which fire constantly
    (e.g. due to faulty hardware).

    Parameters:
        coordinates: list of (x,y) coordinates for which all events will be deleted.
    """

    ordering: str
    coordinates: Optional = None
    hot_pixel_frequency: Optional = None

    def __call__(self, events):
        if self.hot_pixel_frequency:
            self.coordinates = functional.identify_hot_pixel(
                events=events,
                sensor_size=sensor_size,
                ordering=ordering,
                hot_pixel_frequency=self.hot_pixel_frequency,
            )

        print(f"Filtered {len(self.coordinates)} hot pixels.")

        return functional.drop_pixel_numpy(
            events=events, ordering=ordering, coordinates=self.coordinates,
        )


@dataclass
class Downsample:
    """Multiplies timestamps and spatial pixel coordinates with separate factors.
    Useful when the native temporal and/or spatial resolution of the original sensor is too
    high for downstream processing, notably when converting to dense representations of some sort.
    This transform does not drop any events.

    Parameters:
        time_factor (float): value to multiply timestamps with. Default is 0.001.
        spatial_factor (float): value to multiply pixel coordinates with. Default is 1.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    time_factor: float = 1e-3
    spatial_factor: float = 1

    def __call__(self, events):
        events = functional.time_skew_numpy(
            events, self.ordering, coefficient=self.time_factor
        )
        events, sensor_size = functional.spatial_resize_numpy(
            events,
            self.sensor_size,
            self.ordering,
            spatial_factor=self.spatial_factor,
            integer_coordinates=True,
        )
        return events


@dataclass
class NumpyAsType:
    """
    Change dtype of numpy ndarray to custom dtype.
    
    Parameters: 
        dtype: data type that the array should be cast to
    """

    dtype: np.dtype

    def __call__(self, events):
        return events.astype(self.dtype)


@dataclass
class RandomFlipPolarity:
    """Flips polarity of individual events with flip_probability.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        flip_probability (float): probability of flipping individual event polarities
    """

    ordering: str
    flip_probability: float = 0.5

    def __call__(self, events):
        assert "p" in self.ordering
        p_loc = self.ordering.index("p")
        flips = np.ones(len(events))
        probs = np.random.rand(len(events))
        flips[probs < self.flip_probability] = -1
        events[:, p_loc] = events[:, p_loc] * flips
        return events


@dataclass
class RandomFlipLR:
    """Flips events in x. Pixels map as:

        x' = width - x

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    ordering: str
    sensor_size: Tuple[int, int]
    flip_probability: float = 0.5

    def __call__(self, events):
        assert "x" in self.ordering
        if np.random.rand() <= self.flip_probability:
            x_loc = self.ordering.index("x")
            events[:, x_loc] = self.sensor_size[0] - 1 - events[:, x_loc]
        return events


@dataclass
class RandomFlipUD:
    """
    Flips events and images in y. Pixels map as:

        y' = height - y

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    ordering: str
    sensor_size: Tuple[int, int]
    flip_probability: float = 0.5

    def __call__(self, events):
        assert "y" in self.ordering
        if np.random.rand() <= self.flip_probability:
            y_loc = self.ordering.index("y")
            events[:, y_loc] = self.sensor_size[1] - 1 - events[:, y_loc]
        return events


@dataclass
class RandomTimeReversal:
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

    Parameters:
        flip_probability (float): probability of performing the flip
    """

    ordering: str
    flip_probability: float = 0.5

    def __call__(self, events):
        assert "t" and "p" in self.ordering
        if np.random.rand() < self.flip_probability:
            t_loc = self.ordering.index("t")
            p_loc = self.ordering.index("p")
            events[:, t_loc] = np.max(events[:, t_loc]) - events[:, t_loc]
            events[:, p_loc] *= -1
        return events


@dataclass
class RefractoryPeriod:
    """Sets a refractory period for each pixel, during which events will be
    ignored/discarded. We keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}
    
    for each pixel.

    Parameters:
        refractory_period (float): refractory period for each pixel in time unit
    """

    ordering: str
    refractory_period: float

    def __call__(self, events):
        return functional.refractory_period_numpy(
            events, self.ordering, self.refractory_period
        )


@dataclass
class SpatialJitter:
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:

        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        variance_x (float): squared sigma value for the distribution in the x direction
        variance_y (float): squared sigma value for the distribution in the y direction
        sigma_x_y (float): changes skewness of distribution, only change if you want shifts along diagonal axis.
        integer_jitter (bool): when True, x and y coordinates will be shifted by integer rather values instead of floats.
        clip_outliers (bool): when True, events that have been jittered outside the sensor size will be dropped.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    variance_x: float = 1
    variance_y: float = 1
    sigma_x_y: float = 0
    integer_jitter: bool = False
    clip_outliers: bool = False

    def __call__(self, events):
        return functional.spatial_jitter_numpy(
            events=events,
            sensor_size=self.sensor_size,
            ordering=self.ordering,
            variance_x=self.variance_x,
            variance_y=self.variance_y,
            sigma_x_y=self.sigma_x_y,
            integer_jitter=self.integer_jitter,
            clip_outliers=self.clip_outliers,
        )


@dataclass
class TimeJitter:
    """Changes timestamp for each event by drawing samples from a Gaussian
    distribution and adding them to each timestamp.

    Parameters:
        std (float): change the standard deviation of the time jitter
        integer_jitter (bool): will round the jitter that is added to timestamps
        clip_negative (bool): drops events that have negative timestamps
        sort_timestamps (bool): sort the events by timestamps
    """

    ordering: str
    std: float = 1
    integer_jitter: bool = False
    clip_negative: bool = False
    sort_timestamps: bool = False

    def __call__(self, events):
        return functional.time_jitter_numpy(
            events,
            self.ordering,
            self.std,
            self.integer_jitter,
            self.clip_negative,
            self.sort_timestamps,
        )


@dataclass
class TimeSkew:
    """Skew all event timestamps according to a linear transform,
    potentially sampled from a distribution of acceptable functions.

    Parameters:
        coefficient: a real-valued multiplier applied to the timestamps of the events.
                     E.g. a coefficient of 2.0 will double the effective delay between any
                     pair of events.
        offset: value by which the timestamps will be shifted after multiplication by
                the coefficient. Negative offsets are permissible but may result in
                in an exception if timestamps are shifted below 0.
        integer_time: flag that specifies if timestamps should be rounded to
                             nearest integer after skewing.
    """

    ordering: str
    coefficient: float
    offset: float = 0
    integer_time: bool = False

    def __call__(self, events):
        return functional.time_skew_numpy(
            events, self.ordering, self.coefficient, self.offset, self.integer_time
        )


@dataclass
class UniformNoise:
    """Introduces a fixed number of noise events that are uniformly distributed across all provided 
    dimensions, e.g. x, y, t and p.

    Parameters:
        n_noise_events: number of events that are added to the sample.
    """

    ordering: str
    n_noise_events: int

    def __call__(self, events):
        noise_events = []
        for channel in self.ordering:
            channel_index = self.ordering.index(channel)
            event_channel = events[:, channel_index]
            channel_samples = np.random.uniform(
                low=event_channel.min(),
                high=event_channel.max(),
                size=self.n_noise_events,
            )
            noise_events.append(channel_samples)
        noise_events = np.column_stack(noise_events)
        events = np.concatenate((events, noise_events))
        t_index = self.ordering.index("t")
        return events[np.argsort(events[:, t_index]), :]


@dataclass
class ToAveragedTimesurface:
    """Representation that creates averaged timesurfaces for each event for one recording. Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf

    Parameters:
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        time_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    cell_size = 10
    surface_size = 7
    temporal_window = 5e5
    tau = 5e3
    decay = "lin"
    merge_polarities = False

    def __call__(self, events):
        return functional.to_averaged_timesurface(
            events,
            self.sensor_size,
            self.ordering,
            self.cell_size,
            self.surface_size,
            self.temporal_window,
            self.tau,
            self.decay,
            self.merge_polarities,
        )


@dataclass
class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).
    You can set one of the first 4 parameters to choose the slicing method. Depending on which method you choose,
    overlap will assume different functionality, whether that might be temporal overlap, number of events
    or fraction of a bin. As a rule of thumb, here are some considerations if you are unsure which slicing
    method to choose:

    * If your recordings are of roughly the same length, a safe option is to set time_window. Bare in mind
      that the number of events can vary greatly from slice to slice, but will give you some consistency when
      training RNNs or other algorithms that have time steps.

    * If your recordings have roughly the same amount of activity / number of events and you are more interested
      in the spatial composition, then setting spike_count will give you frames that are visually more consistent.

    * The previous time_window and spike_count methods will likely result in a different amount of frames for each
      recording. If your training method benefits from consistent number of frames across a dataset (for easier
      batching for example), or you want a parameter that is easier to set than the exact window length or number
      of events per slice, consider fixing the number of frames by setting n_time_bins or n_event_bins. The two
      methods slightly differ with respect to how the slices are distributed across the recording. You can define
      an overlap between 0 and 1 to provide some robustness.

    Parameters:
        time_window (float): time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        spike_count (int): number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): overlap between frames defined either in time units, number of events or number of bins between 0 and 1.
        include_incomplete (bool): if True, includes overhang slice when time_window or spike_count is specified.
                                   Not valid for bin_count methods.
        merge_polarities (bool): if True, merge polarity channels to a single channel.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    time_window: Optional[float] = None
    spike_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False
    merge_polarities: bool = False

    def __call__(self, events):
        return functional.to_frame_numpy(
            events=events,
            sensor_size=self.sensor_size,
            ordering=self.ordering,
            time_window=self.time_window,
            spike_count=self.spike_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
            merge_polarities=self.merge_polarities,
        )


@dataclass
class ToSparseTensor:
    """Turn event array (N,E) into sparse Tensor (B,T,W,H) if E is 4 (mostly event camera recordings),
    otherwise into sparse tensor (B,T,W) for mostly audio recordings.
    
    Parameters:
        backend (str): choose which framework to use. Default is pytorch, other possibilities are tensorflow and scipy.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    backend: str = "pytorch"
    merge_polarities: bool = False

    def __call__(self, events):
        if self.backend == "pytorch" or "pt" or "torch":
            tensor = functional.to_sparse_tensor_pytorch(
                events=events,
                sensor_size=self.sensor_size,
                ordering=self.ordering,
                merge_polarities=self.merge_polarities,
            )
        elif self.backend == "tensorflow" or "tf":
            tensor = functional.to_sparse_tensor_tensorflow(
                events=events,
                sensor_size=self.sensor_size,
                ordering=self.ordering,
                merge_polarities=self.merge_polarities,
            )
        else:
            raise NotImplementedError
        return tensor


@dataclass
class ToDenseTensor:
    """Creates dense representation of events.
    
    Parameters:
        backend (str): choose which framework to use. Default is pytorch, alternatively tensorflow.
    """

    ordering: str
    sensor_size: Tuple[int, int]
    backend: str = "pytorch"
    merge_polarities: bool = False

    def __call__(self, events):
        tensor = ToSparseTensor(ordering=self.ordering, sensor_size=self.sensor_size, backend=self.backend, merge_polarities=self.merge_polarities)(events)
        if self.backend == "pytorch" or "pt" or "torch":
            return tensor.to_dense()
            

@dataclass
class ToTimesurface:
    """Representation that creates timesurfaces for each event in the recording. Modeled after the paper
    Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476

    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.
    """
    
    ordering: str
    sensor_size: Tuple[int, int]
    surface_dimensions: Tuple[int, int] = (7, 7)
    tau: float = 5e3
    decay: str = "lin"
    merge_polarities: bool = False

    def __call__(self, events):
        return functional.to_timesurface_numpy(
            events,
            sensor_size=self.sensor_size,
            ordering=self.ordering,
            surface_dimensions=self.surface_dimensions,
            tau=self.tau,
            decay=self.decay,
            merge_polarities=self.merge_polarities,
        )


@dataclass
class ToVoxelGrid:
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    
    Parameters:
        n_time_bins (int): fixed number of time bins to slice the event sample into."""

    ordering: str
    sensor_size: Tuple[int, int]
    n_time_bins: int

    def __call__(self, events):
        return functional.to_voxel_grid_numpy(
            events, self.sensor_size, self.ordering, self.n_time_bins
        )


@dataclass
class Repeat:
    """Copies target n times. Useful to transform sample labels into sequences."""

    n_repeat: int
    
    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.n_repeat, 1])


@dataclass
class ToOneHotEncoding:
    """Transforms one or more targets into a one hot encoding scheme."""

    n_classes: int
    
    def __call__(self, target):
        return np.eye(self.n_classes)[target]
