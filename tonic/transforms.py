from dataclasses import dataclass
from . import functional
from typing import Callable, Optional, Tuple, Union
import numpy as np


class Compose:
    """Composes several transforms together. This a literal copy of torchvision.transforms.Compose function for convenience.

    Parameters:
        transforms (list of ``Transform`` objects): list of transform(s) to compose.
                                                    Can combine Tonic, PyTorch Vision/Audio transforms.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Denoise(filter_time=10000),
        >>>     transforms.ToFrame(n_time_bins=3),
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


@dataclass(frozen=True)
class Denoise:
    """Drops events that are 'not sufficiently connected to other events in the recording.'
    In practise that means that an event is dropped if no other event occured within a spatial neighbourhood
    of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to filter noisy recordings
    where events occur isolated in time.

    Parameters:
        filter_time (float): maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.
    """

    filter_time: float = 10000

    def __call__(self, events):
        return functional.denoise_numpy(events=events, filter_time=self.filter_time)


@dataclass(frozen=True)
class DropEvent:
    """Randomly drops events with p.

    Parameters:
        p (float): probability of dropping out event.
        random_p (bool): randomize the dropout probability
                                 between 0 and p.
    """

    p: float = 0.5
    random_p: bool = False

    def __call__(self, events):

        return functional.drop_event_numpy(events, self.p, self.random_p)


@dataclass
class DropPixel:
    """Drops events for individual pixels. If the locations of pixels to be dropped is known, a
    list of x/y coordinates can be passed directly. Alternatively, a cutoff frequency for each pixel can be defined
    above which pixels will be deactivated completely. This prevents so-called *hot pixels* which fire constantly
    (e.g. due to faulty hardware).

    Parameters:
        coordinates: list of (x,y) coordinates for which all events will be deleted.
        hot_pixel_frequency: drop pixels completely that fire higher than the given frequency.
    """

    coordinates: Optional[Tuple] = None
    hot_pixel_frequency: Optional[int] = None

    def __call__(self, events):

        if events.dtype.names is not None:
            # assert "x", "y", "p" in events.dtype.names
            if self.hot_pixel_frequency:
                self.coordinates = functional.identify_hot_pixel(
                    events=events, hot_pixel_frequency=self.hot_pixel_frequency
                )

            return functional.drop_pixel_numpy(
                events=events, coordinates=self.coordinates
            )

        elif len(events.shape) == 4 or len(events.shape) == 3:
            if self.hot_pixel_frequency:
                self.coordinates = functional.identify_hot_pixel_raster(
                    events=events, hot_pixel_frequency=self.hot_pixel_frequency
                )

            return functional.drop_pixel.drop_pixel_raster(events, self.coordinates)


@dataclass(frozen=True)
class Downsample:
    """Multiplies timestamps and spatial pixel coordinates with separate factors.
    Useful when the native temporal and/or spatial resolution of the original sensor is too
    high for downstream processing, notably when converting to dense representations of some sort.
    This transform does not drop any events.

    Parameters:
        time_factor (float): value to multiply timestamps with. Default is 0.001.
        spatial_factor (float): value to multiply pixel coordinates with. Default is 1.
    """

    time_factor: float = 1e-3
    spatial_factor: float = 1

    def __call__(self, events):
        events = events.copy()
        events = functional.time_skew_numpy(events, coefficient=self.time_factor)
        if "x" in events.dtype.names:
            events["x"] = events["x"] * self.spatial_factor
        if "y" in events.dtype.names:
            events["y"] = events["y"] * self.spatial_factor
        return events


@dataclass(frozen=True)
class MergePolarities:
    """
    After this transform there is only a single polarity left which is zero.
    """

    def __call__(self, events):
        events = events.copy()
        events["p"] = np.zeros_like(events["p"])
        return events


@dataclass(frozen=True)
class NumpyAsType:
    """
    Change dtype of numpy ndarray to custom dtype.

    Parameters:
        dtype: data type that the array should be cast to
    """

    dtype: np.dtype

    def __call__(self, events):
        source_is_structured_array = (
            hasattr(events.dtype, "names") and events.dtype.names != None
        )
        target_is_structured_array = (
            hasattr(self.dtype, "names") and self.dtype.names != None
        )
        if source_is_structured_array and not target_is_structured_array:
            return np.lib.recfunctions.structured_to_unstructured(events, self.dtype)
        elif source_is_structured_array and target_is_structured_array:
            return NotImplementedError
        elif not target_is_structured_array and not source_is_structured_array:
            return events.astype(self.dtype)


@dataclass(frozen=True)
class RandomCrop:
    """Crops the sensor size to a smaller sensor in a random location.

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        target_size: a tuple of x,y target sensor size
    """

    sensor_size: Tuple[int, int, int]
    target_size: Tuple[int, int]

    def __call__(self, events):

        return functional.crop_numpy(
            events=events, sensor_size=self.sensor_size, target_size=self.target_size
        )


@dataclass(frozen=True)
class RandomFlipPolarity:
    """Flips polarity of individual events with p.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        p (float): probability of flipping individual event polarities
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "p" in events.dtype.names
        if np.random.rand() <= self.p:
            events["p"] = np.invert(events["p"].astype(bool)).astype(events.dtype["p"])
        return events


@dataclass(frozen=True)
class RandomFlipLR:
    """Flips events in x. Pixels map as:

        x' = width - x

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip
    """

    sensor_size: Tuple[int, int, int]
    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "x" in events.dtype.names
        if np.random.rand() <= self.p:
            events["x"] = self.sensor_size[0] - 1 - events["x"]
        return events


@dataclass(frozen=True)
class RandomFlipUD:
    """
    Flips events and images in y. Pixels map as:

        y' = height - y

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip
    """

    sensor_size: Tuple[int, int, int]
    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "y" in events.dtype.names
        if np.random.rand() <= self.p:
            events["y"] = self.sensor_size[1] - 1 - events["y"]
        return events


@dataclass(frozen=True)
class RandomTimeReversal:
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

    Parameters:
        p (float): probability of performing the flip
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "t" and "p" in events.dtype.names
        if np.random.rand() < self.p:
            events["t"] = np.max(events["t"]) - events["t"]
            events["p"] *= -1
        return events


@dataclass(frozen=True)
class RefractoryPeriod:
    """Sets a refractory period for each pixel, during which events will be
    ignored/discarded. We keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}

    for each pixel.

    Parameters:
        refractory_period (float): refractory period for each pixel in time unit
    """

    refractory_period: float

    def __call__(self, events):
        return functional.refractory_period_numpy(events, self.refractory_period)


@dataclass(frozen=True)
class SpatialJitter:
    """Changes position for each pixel by drawing samples from a multivariate
    Gaussian distribution with the following properties:

        mean = [x,y]
        covariance matrix = [[variance_x, sigma_x_y],[sigma_x_y, variance_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        variance_x (float): squared sigma value for the distribution in the x direction
        variance_y (float): squared sigma value for the distribution in the y direction
        sigma_x_y (float): changes skewness of distribution, only change if you want shifts along diagonal axis.
        clip_outliers (bool): when True, events that have been jittered outside the sensor size will be dropped.
    """

    sensor_size: Tuple[int, int, int]
    variance_x: float = 1
    variance_y: float = 1
    sigma_x_y: float = 0
    clip_outliers: bool = False

    def __call__(self, events):
        events = events.copy()
        return functional.spatial_jitter_numpy(
            events=events,
            sensor_size=self.sensor_size,
            variance_x=self.variance_x,
            variance_y=self.variance_y,
            sigma_x_y=self.sigma_x_y,
            clip_outliers=self.clip_outliers,
        )


@dataclass
class TimeAlignment:
    """Removes offset for timestamps, so that first events starts at time zero."""

    def __call__(self, events):
        events = events.copy()
        assert "t" in events.dtype.names
        events["t"] -= min(events["t"])
        return events


@dataclass(frozen=True)
class TimeJitter:
    """Changes timestamp for each event by drawing samples from a Gaussian
    distribution and adding them to each timestamp.

    Parameters:
        std (float): change the standard deviation of the time jitter
        clip_negative (bool): drops events that have negative timestamps
        sort_timestamps (bool): sort the events by timestamps after jitter
    """

    std: float = 1
    clip_negative: bool = False
    sort_timestamps: bool = False

    def __call__(self, events):
        events = events.copy()
        return functional.time_jitter_numpy(
            events, self.std, self.clip_negative, self.sort_timestamps
        )


@dataclass(frozen=True)
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
    """

    coefficient: float
    offset: float = 0

    def __call__(self, events):
        events = events.copy()
        return functional.time_skew_numpy(events, self.coefficient, self.offset)


@dataclass(frozen=True)
class UniformNoise:
    """Introduces a fixed number of noise events that are uniformly distributed across event dimensions, e.g. x, y, t and p.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        n_noise_events: number of events that are added to the sample.
    """

    sensor_size: Tuple[int, int, int]
    n_noise_events: int

    def __call__(self, events):

        noise_events = np.zeros(self.n_noise_events, dtype=events.dtype)
        for channel in events.dtype.names:
            event_channel = events[channel]
            if channel == "x":
                low, high = 0, self.sensor_size[0]
            if channel == "y":
                low, high = 0, self.sensor_size[1]
            if channel == "p":
                low, high = 0, self.sensor_size[2]
            if channel == "t":
                low, high = events["t"].min(), events["t"].max()
            noise_events[channel] = np.random.uniform(
                low=low, high=high, size=self.n_noise_events
            )
        events = np.concatenate((events, noise_events))
        return events[np.argsort(events["t"])]


@dataclass(frozen=True)
class ToAveragedTimesurface:
    """Representation that creates averaged timesurfaces for each event for one recording. Taken from the paper
    Sironi et al. 2018, HATS: Histograms of averaged time surfaces for robust event-based object classification
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        temporal_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
        num_workers (int): number of workers to be deployed on the histograms computation. When >1, joblib is required. 
    """

    sensor_size: Tuple[int, int, int]
    cell_size: int = 10
    surface_size: int = 7
    temporal_window: int = 5e5
    tau: int = 5e3
    decay: str = "lin"
    num_workers: int = 1

    def __call__(self, events):
        return functional.to_averaged_timesurface(
            events,
            sensor_size=self.sensor_size,
            cell_size=self.cell_size,
            surface_size=self.surface_size,
            temporal_window=self.temporal_window,
            tau=self.tau,
            decay=self.decay,
            num_workers=self.num_workers,
        )


@dataclass(frozen=True)
class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window),
    constant number of events (spike_count) or constant number of frames (n_time_bins / n_event_bins).
    All the events in one slice are added up in a frame for each polarity.
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
        sensor_size: a 3-tuple of x,y,p for sensor_size
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
    """

    sensor_size: Tuple[int, int, int]
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False

    def __call__(self, events):

        return functional.to_frame_numpy(
            events=events,
            sensor_size=self.sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )


@dataclass(frozen=True)
class ToImage:
    """Counts up all events to a *single* image of size sensor_size. ToImage will typically
    be used in combination with SlicedDataset to cut a recording into smaller chunks that 
    are then individually binned to frames. 
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):

        frames = functional.to_frame_numpy(
            events=events, sensor_size=self.sensor_size, event_count=len(events)
        )

        return frames.squeeze(0)


@dataclass(frozen=True)
class ToTimesurface:
    """Representation that creates timesurfaces for each event in the recording. Modeled after the paper
    Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    """

    sensor_size: Tuple[int, int, int]
    surface_dimensions: Union[None, Tuple[int, int]] = None
    tau: float = 5e3
    decay: str = "lin"

    def __call__(self, events):

        return functional.to_timesurface_numpy(
            events=events,
            sensor_size=self.sensor_size,
            surface_dimensions=self.surface_dimensions,
            tau=self.tau,
            decay=self.decay,
        )


@dataclass(frozen=True)
class ToVoxelGrid:
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical flow, depth, and egomotion

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        n_time_bins (int): fixed number of time bins to slice the event sample into."""

    sensor_size: Tuple[int, int, int]
    n_time_bins: int

    def __call__(self, events):

        return functional.to_voxel_grid_numpy(
            events.copy(), self.sensor_size, self.n_time_bins
        )


@dataclass(frozen=True)
class Repeat:
    """Copies target n times. Useful to transform sample labels into sequences."""

    n_repeat: int

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.n_repeat, 1])


@dataclass(frozen=True)
class ToOneHotEncoding:
    """Transforms one or more targets into a one hot encoding scheme."""

    n_classes: int

    def __call__(self, target):
        return np.eye(self.n_classes)[target]
