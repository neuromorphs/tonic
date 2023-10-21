import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from . import functional


class Compose:
    """Composes several transforms together. This a literal copy of torchvision.transforms.Compose
    function for convenience.

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
            if len(events) == 0:
                break
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
class CenterCrop:
    """Crops events at the center to a specific output size. If output size is smaller than input
    sensor size along any dimension, padding will be used, which doesn't influence the number of
    events on that axis but just their spatial location after cropping. Make sure to use the
    cropped sensor size for any transform after CenterCrop.

    Parameters:
        sensor_size (tuple): Size of the sensor that was used [W,H,P]
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    sensor_size: Tuple[int, int, int]
    size: Union[int, Tuple[int, int]]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if type(self.size) == int:
            self.size = [self.size, self.size]
        offsets = (self.sensor_size[0] - self.size[0]) // 2, (
            self.sensor_size[1] - self.size[1]
        ) // 2
        offset_idx = [max(offset, 0) for offset in offsets]
        cropped_events = events[
            (offset_idx[0] <= events["x"])
            & (events["x"] < (offset_idx[0] + self.size[0]))
            & (offset_idx[1] <= events["y"])
            & (events["y"] < (offset_idx[1] + self.size[1]))
        ]
        cropped_events["x"] -= offsets[0]
        cropped_events["y"] -= offsets[1]
        return cropped_events


@dataclass
class CropTime:
    """Drops events with timestamps below min and above max.

    Parameters:
        min (int): The minimum timestamp below which all events are dropped. Zero by default.
        max (int): The maximum timestamp above which all events are dropped.

    Example:
        >>> transform = tonic.transforms.CropTime(min=1000, max=20000)
    """

    min: int = 0
    max: int = None

    def __call__(self, events):
        assert "t" in events.dtype.names
        if self.max is None:
            self.max = np.max(events["t"])
        return events[(events["t"] >= self.min) & (events["t"] <= self.max)]


@dataclass(frozen=True)
class Denoise:
    """Drops events that are spatio-temporally not sufficiently close enough to other events in the
    sample. In practise that means that an event is dropped if no other event occured within a
    spatial neighbourhood of 1 pixel and a temporal neighbourhood of filter_time time units. Useful
    to filter noisy recordings where events occur isolated in time.

    Parameters:
        filter_time (float): minimum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less output events.
                    Use same unit of time as the events have.

    Example:
        >>> transform = tonic.transforms.Denoise(filter_time=10000)
    """

    filter_time: float

    def __call__(self, events):
        return functional.denoise_numpy(events=events, filter_time=self.filter_time)


@dataclass
class Decimation:
    """Deterministically drops every nth event for every spatial location x (and potentially y).

    Parameters:
        n (int): The event stream for each x/y location is reduced to 1/n.

    Example:
        >>> transform = tonic.transforms.Decimation(n=5)
    """

    n: int

    def __call__(self, events):
        return functional.decimate_numpy(events=events, n=self.n)


@dataclass(frozen=True)
class DropEvent:
    """Randomly drops events with probability p. If random_p is selected, the drop probability is
    randomized between 0 and p.

    Parameters:
        p (float or tuple of floats): Probability of dropping events. Can be a tuple of floats (p_min, p_max), so that p is sampled from the range.

    Example:
        >>> transform1 = tonic.transforms.DropEvent(p=0.2)
        >>> transform2 = tonic.transforms.DropEvent(p=(0, 0.5))
    """

    p: Union[float, Tuple[float, float]]

    @staticmethod
    def get_params(p: Union[float, Tuple[float, float]]):
        if type(p) == tuple:
            p = (p[1] - p[0]) * np.random.random_sample() + p[0]
        return p

    def __call__(self, events):
        p = self.get_params(p=self.p)
        return functional.drop_event_numpy(events=events, drop_probability=p)


@dataclass(frozen=True)
class DropEventByTime:
    """Drops events in a certain time interval with a length proportional to a specified ratio of
    the original length.

    Parameters:
        duration_ratio (Union[float, Tuple[float]], optional): the length of the dropped time interval, expressed in a ratio of the original sequence duration.
            - If a float, the value is used to calculate the interval length
            - If a tuple of 2 floats, the ratio is randomly chosen in [min, max)
            Defaults to 0.2.

    Example:
        >>> transform = tonic.transforms.DropEventByTime(duration_ratio=(0.1, 0.8))
    """

    duration_ratio: Union[float, Tuple[float, float]] = 0.2

    def __call__(self, events):
        return functional.drop_by_time_numpy(events, self.duration_ratio)


@dataclass(frozen=True)
class DropEventByArea:
    """Drops events located in a randomly chosen box area. The size of the box area is defined by a
    specified ratio of the sensor size.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]
        area_ratio (Union[float, Tuple[float]], optional): Ratio of the sensor resolution that determines the size of the box area where events are dropped.
            - if a float, the value is used to calculate the size of the box area
            - if a tuple of 2 floats, the ratio is randomly chosen in [min, max)
            Defaults to 0.2.

    Example:
        >>> transform = tonic.transforms.DropEventByArea(sensor_size=(128,128,2), area_ratio=(0.1, 0.8))
    """

    sensor_size: Tuple[int, int, int]
    area_ratio: Union[float, Tuple[float, float]] = 0.2

    def __call__(self, events):
        return functional.drop_by_area_numpy(events, self.sensor_size, self.area_ratio)


@dataclass
class DropPixel:
    """Drops events for individual pixels. If the locations of pixels to be dropped is known, a
    list of x/y coordinates can be passed directly. Alternatively, a cutoff frequency for each
    pixel can be defined above which pixels will be deactivated completely. This prevents so-
    called *hot pixels* which fire at a high frequency even in the absence of any input signal
    (e.g. due to faulty hardware).

    Parameters:
        coordinates: List of (x,y) coordinates for which all events will be deleted.
        hot_pixel_frequency: Drop pixels completely that fire higher than the given frequency.

    Example:
        >>> from tonic.transforms import DropPixel
        >>> transform1 = DropPixel(coordinates=[[10,10], [10,11], [11,10], [11,11]])
        >>> transform2 = DropPixel(hot_pixel_frequency=60) # Hertz
    """

    coordinates: Optional[List[Tuple[int, int]]] = None
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

            return functional.drop_pixel_raster(events, self.coordinates)


@dataclass(frozen=True)
class Downsample:
    """Multiplies timestamps and spatial pixel coordinates with separate factors. Useful when the
    native temporal and/or spatial resolution of the original sensor is too high for downstream
    processing, notably when converting to dense representations of some sort. This transform does
    not drop any events.

    Parameters:
        time_factor (float): value to multiply timestamps with. Default is 1.
        spatial_factor (float or tuple of floats): values to multiply pixel coordinates with. Default is 1.
                                                   Note that when using subsequential transforms that require
                                                   sensor_size, you must change the spatial values for the later
                                                   transformation.
        sensor_size (tuple): size of the sensor that was used [W,H,P]
        target_size (tuple): size of the desired resolution [W,H]

    Example:
        >>> from tonic.transforms import Downsample
        >>> transform1 = Downsample(time_factor=0.001) # change us to ms
        >>> transform2 = Downsample(spatial_factor=0.25) # reduce focal plane to 1/4.
        >>> transform3 = Downsample(sensor_size=(40, 20, 2), target_size=(10, 5)) # reduce focal plane to 1/4.
    """

    time_factor: float = 1
    spatial_factor: Union[float, Tuple[float, float]] = 1
    sensor_size: Optional[Tuple[int, int, int]] = None
    target_size: Optional[Tuple[int, int]] = None

    @staticmethod
    def get_params(spatial_factor: Union[int, Tuple[int, int]]):
        if not type(spatial_factor) == tuple:
            spatial_factor = (spatial_factor, spatial_factor)
        return spatial_factor

    def __call__(self, events):
        events = events.copy()

        if self.target_size is not None:
            # Ensure sensor_size is not None when target_size is not None
            assert self.sensor_size is not None
            # If both target_size and spatial_factor declared, override spatial_factor value in argument
            spatial_factor = np.asarray(self.target_size) / self.sensor_size[:-1]
        else:
            spatial_factor = self.get_params(spatial_factor=self.spatial_factor)

        events = functional.time_skew_numpy(events, coefficient=self.time_factor)
        if "x" in events.dtype.names:
            events["x"] = events["x"] * spatial_factor[0]
        if "y" in events.dtype.names:
            events["y"] = events["y"] * spatial_factor[1]
        return events


@dataclass(frozen=True)
class EventDrop:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]

    Example:
        >>> transform = tonic.transforms.EventDrop(sensor_size=(128,128,2))
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        choice = np.random.randint(0, 4)
        if choice == 0:
            return events
        if choice == 1:
            duration_ratio = np.random.randint(1, 10) / 10.0
            return functional.drop_by_time_numpy(events, duration_ratio)
        if choice == 2:
            area_ratio = np.random.randint(1, 6) / 20.0
            return functional.drop_by_area_numpy(events, self.sensor_size, area_ratio)
        if choice == 3:
            ratio = np.random.randint(1, 10) / 10.0
            return functional.drop_event_numpy(events, ratio)


@dataclass(frozen=True)
class EventDownsampling:
    """Applies EventDownsampling from the paper "Insect-inspired Spatio-temporal Downsampling of Event-based Input."
        Allows:
            1. Integrator based method to perform spatio-temporal event-based downsampling
            2. Differentiator based method to perform spatio-temporal event-based downsampling

    Parameters:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]
        target_size (Tuple): size of the desired resolution [W,H]
        dt (float): temporal resolution of events in ms
        downsampling_method (str): string stating downsampling method. Choose from ['naive', 'integrator', 'differentiator']
        noise_threshold (int): set number of events in downsampled pixel required to emit spike. Zero by default.
        differentiator_time_bins (int): number of differentiator time bins within dt. Two by default.

    Example:
        >>> transform1 = tonic.transforms.EventDownsampling(sensor_size=(640,480,2), target_size=(20,15), dt=0.5,
                                                           downsampling_method='integrator')
        >>> transform2 = tonic.transforms.EventDownsampling(sensor_size=(640,480,2), target_size=(20,15), dt=0.5,
                                                           downsampling_method='differentiator', noise_threshold=2,
                                                           differentiator_time_bins=3)
    """

    sensor_size: Tuple[int, int, int]
    target_size: Tuple[int, int]
    downsampling_method: str
    dt: Optional[float] = None
    noise_threshold: Optional[int] = None
    differentiator_time_bins: Optional[int] = None

    def __call__(self, events):
        assert self.downsampling_method in ["integrator", "differentiator"]

        if self.downsampling_method == "integrator":
            return functional.integrator_downsample(
                events=events,
                sensor_size=self.sensor_size,
                target_size=self.target_size,
                dt=self.dt,
                noise_threshold=self.noise_threshold,
            )

        elif self.downsampling_method == "differentiator":
            return functional.differentiator_downsample(
                events=events,
                sensor_size=self.sensor_size,
                target_size=self.target_size,
                dt=self.dt,
                noise_threshold=self.noise_threshold,
                differentiator_time_bins=self.differentiator_time_bins,
            )


@dataclass(frozen=True)
class MergePolarities:
    """Sets all polarities to zero. This transform does not have any parameters.

    Example:
        >>> transform = tonic.transforms.MergePolarities()
    """

    def __call__(self, events):
        events = events.copy()
        events["p"] = np.zeros_like(events["p"])
        return events


@dataclass(frozen=True)
class RandomCrop:
    """Crops the sensor size to a smaller size in a random location.

    x' = x - new_sensor_start_x

    y' = y - new_sensor_start_y

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        target_size: a tuple of x,y target sensor size

    Example:
        >>> transform = tonic.transforms.RandomCrop(sensor_size=(340, 240, 2), target_size=(50, 50))
    """

    sensor_size: Tuple[int, int, int]
    target_size: Tuple[int, int]

    def __call__(self, events):
        return functional.crop_numpy(
            events=events, sensor_size=self.sensor_size, target_size=self.target_size
        )


@dataclass
class RandomDropPixel:
    """Drops all events for individual pixels with a given probability.

    Parameters:
        p: Probability of pixel being dropped. Stochastic transform.
        sensor_size: a 3-tuple of x,y,p for sensor_size. Not necessary when RandomDropPixel is applied to rasters.

    Example:
        >>> from tonic.transforms import RandomDropPixel
        >>> transform = DropPixel(p=0.2)
    """

    p: float
    sensor_size: Optional[Tuple[int, int, int]] = None

    def __call__(self, events):
        if events.dtype.names is not None:
            if self.sensor_size is None:
                sensor_size_x, sensor_size_y, _ = int(events["x"].max() + 1), int(
                    events["y"].max() + 1
                )
            else:
                sensor_size_x, sensor_size_y, _ = self.sensor_size

            coordinates_x, coordinates_y = np.where(
                np.random.rand(sensor_size_x, sensor_size_y) < self.p
            )
            coordinates = list(zip(coordinates_x, coordinates_y))
            return functional.drop_pixel_numpy(events=events, coordinates=coordinates)

        elif len(events.shape) == 4 or len(events.shape) == 3:
            sensor_size_y, sensor_size_x = events.shape[-2:]
            coordinates_x, coordinates_y = np.where(
                np.random.rand(sensor_size_x, sensor_size_y) < self.p
            )
            coordinates = list(zip(coordinates_x, coordinates_y))
            return functional.drop_pixel_raster(events, coordinates)


@dataclass(frozen=True)
class RandomFlipPolarity:
    """Flips polarity of individual events with p. Changes polarities 1 to 0 and polarities [-1, 0]
    to 1.

    Parameters:
        p (float): probability of flipping individual event polarities

    Example:
        >>> transform = tonic.transforms.RandomFlipPolarity(p=0.3)
    """

    p: float = 0.5

    def __post_init__(self):
        assert 0 <= self.p <= 1

    def __call__(self, events):
        events = events.copy()
        assert "p" in events.dtype.names
        if np.random.rand() <= self.p:
            events["p"] = np.invert(events["p"].astype(bool)).astype(events.dtype["p"])
        return events


@dataclass(frozen=True)
class RandomFlipLR:
    """Flips events in x with probability p. Pixels map as:

        x' = width - x

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip

    Example:
        >>> transform = tonic.transforms.RandomFlipLR(p=0.3)
    """

    sensor_size: Tuple[int, int, int]
    p: float = 0.5

    def __post_init__(self):
        assert 0 <= self.p <= 1

    def __call__(self, events):
        events = events.copy()
        assert "x" in events.dtype.names
        if np.random.rand() <= self.p:
            events["x"] = self.sensor_size[0] - 1 - events["x"]
        return events


@dataclass(frozen=True)
class RandomFlipUD:
    """Flips events in y with probability p. Pixels map as:

        y' = height - y

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip

    Example:
        >>> transform = tonic.transforms.RandomFlipUD(p=0.3)
    """

    sensor_size: Tuple[int, int, int]
    p: float = 0.5

    def __post_init__(self):
        assert 0 <= self.p <= 1

    def __call__(self, events):
        events = events.copy()
        assert "y" in events.dtype.names
        if np.random.rand() <= self.p:
            events["y"] = self.sensor_size[1] - 1 - events["y"]
        return events


@dataclass(frozen=True)
class RandomTimeReversal:
    """Reverses temporal order of events with probability p.

        .. math::
           t_i' = max(t) - t_i

    Parameters:
        p (float): probability of performing the flip
        flip_polarities (bool): if the time is reversed, also flip the polarities. True by default.

    Example:
        >>> transform = tonic.transforms.RandomTimeReversal(p=0.3)
    """

    p: float = 0.5
    flip_polarities: bool = True

    def __post_init__(self):
        assert 0 <= self.p <= 1

    def __call__(self, events):
        events = events.copy()

        if np.random.rand() < self.p:
            # if events is a raster-like numpy array in shape [t, p, h, w] or [t, p, x]
            if events.ndim == 4 or events.ndim == 3:
                # reverse both time and polarity
                # array with negative strides are not supported to be converted to tensor by torch, so return a copy
                return events[::-1, ::-1, ...].copy()

            assert "t" and "p" in events.dtype.names
            events["t"] = np.max(events["t"]) - events["t"]
            if self.flip_polarities:
                events["p"] = np.invert(events["p"].astype(bool)).astype(
                    events.dtype["p"]
                )
            events = events[::-1]
        return events


@dataclass(frozen=True)
class RefractoryPeriod:
    """Sets a refractory period for each pixel, during which events will be ignored/discarded. We
    keep events if:

        .. math::
            t_n - t_{n-1} > t_{refrac}

    for each pixel.

    Parameters:
        delta (int): Refractory period for each pixel. Use same time
                     unit as event timestamps. Can use a 2-tuple to
                     sample from a range.

    >>> transform1 = tonic.transforms.RefractoryPeriod(delta=1000)
    >>> transform2 = tonic.transforms.RefractoryPeriod(delta=[0, 1000])
    """

    delta: Union[int, Tuple[int, int]]

    @staticmethod
    def get_params(delta: Union[int, Tuple[int, int]]):
        if type(delta) == tuple:
            delta = int((delta[1] - delta[0]) * np.random.random_sample() + delta[0])
        return delta

    def __call__(self, events):
        delta = self.get_params(delta=self.delta)
        return functional.refractory_period_numpy(
            events=events, refractory_period=delta
        )


@dataclass(frozen=True)
class SpatialJitter:
    """Changes x/y coordinate for each event by adding samples from a multivariate Gaussian
    distribution. It with the following properties:

        .. math::
            mean = [x,y]

            \Sigma = [[var_x, sigma_{xy}],[sigma_{xy}, var_y]]

    Jittered events that lie outside the focal plane will be dropped if clip_outliers is True.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        var_x (float): variance for the distribution in the x direction
        var_y (float): variance for the distribution in the y direction
        sigma_xy (float): changes skewness of distribution, only change if you want shifts along diagonal axis.
        clip_outliers (bool): when True, events that have been jittered outside the sensor size will be dropped.
    """

    sensor_size: Tuple[int, int, int]
    var_x: float = 1
    var_y: float = 1
    sigma_xy: float = 0
    clip_outliers: bool = False

    def __call__(self, events):
        events = events.copy()
        return functional.spatial_jitter_numpy(
            events=events,
            sensor_size=self.sensor_size,
            var_x=self.var_x,
            var_y=self.var_y,
            sigma_xy=self.sigma_xy,
            clip_outliers=self.clip_outliers,
        )


@dataclass
class TimeAlignment:
    """Removes offset for timestamps, so that first event starts at time zero."""

    def __call__(self, events):
        events = events.copy()
        assert "t" in events.dtype.names
        events["t"] -= min(events["t"])
        return events


@dataclass(frozen=True)
class TimeJitter:
    """Changes timestamp for each event by adding samples from a Gaussian distribution.

    Parameters:
        std (sequence or float): the standard deviation of the time jitter.
        clip_negative (bool): drops events that have negative timestamps.
        sort_timestamps (bool): sort the events by timestamps after jitter.
    """

    std: float
    clip_negative: bool = True
    sort_timestamps: bool = False

    def __call__(self, events):
        events = events.copy()
        return functional.time_jitter_numpy(
            events, self.std, self.clip_negative, self.sort_timestamps
        )


@dataclass(frozen=True)
class TimeSkew:
    """Skew all event timestamps according to a linear transform.

    Parameters:
        coefficient: a real-valued multiplier applied to the timestamps of the events.
                     E.g. a coefficient of 2.0 will double the effective delay between any
                     pair of events. Can provide a tuple for a range of values.
        offset: value by which the timestamps will be shifted after multiplication by
                the coefficient. Negative offsets are permissible but may result in
                in an exception if timestamps are shifted below 0. Tuple of values might
                be provided as a range to sample from.

    Example:
        >>> transform1 = TimeSkew(coefficient=1.3, offset=100)
        >>> transform2 = TimeSkew(coefficient=[0.8, 1.2], offset=[0, 150])
    """

    coefficient: Union[float, Tuple[float, float]]
    offset: Union[float, Tuple[float, float]] = 0

    def __call__(self, events):
        events = events.copy()
        return functional.time_skew_numpy(events, self.coefficient, self.offset)


@dataclass(frozen=True)
class UniformNoise:
    """Adds a fixed number of n noise events that are uniformly distributed across sensor size
    dimensions such as x, y, t and p.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        n: Number of events that are added. Can be a tuple of integers,
           so that n is sampled from a range.

    Example:
        >>> transform = tonic.transforms.UniformNoise(sensor_size=(340, 240, 2), n=3000)
    """

    sensor_size: Tuple[int, int, int]
    n: Union[int, Tuple[int, int]]

    @staticmethod
    def get_params(n: Union[int, Tuple[int, int]]):
        if type(n) == tuple:
            n = int((n[1] - n[0]) * np.random.random_sample() + n[0])
        return n

    def __call__(self, events):
        n = self.get_params(n=self.n)
        return functional.uniform_noise_numpy(
            events=events, sensor_size=self.sensor_size, n=n
        )


@dataclass(frozen=True)
class NumpyAsType:
    """Change dtype of numpy ndarray to custom dtype. This transform is necessary for example if
    you want to load raw events using a PyTorch dataloader. The original events coming from any
    dataset in Tonic are structured numpy arrays, so that they can be indexed as events["t"] or
    events["p"] etc. Pytorch's dataloader however does not support the conversion from structured
    numpy arrays to Tensors, that's why we need to employ at least NumpyAsType(int) to convert the
    structured array into an unstructured one before handing it to the dataloader.

    Parameters:
        dtype: data type that the array should be cast to.

    Example:
        >>> # indexing the dataset directly provides structured numpy arrays
        >>> dataset = tonic.datasets.NMNIST(save_to='data')
        >>> events, targets = dataset[100]
        >>>
        >>> # this doesn't work
        >>> dataloader = torch.utils.data.DataLoader(dataset)
        >>> events, targets = next(iter(dataloader))
        >>>
        >>> # we need to convert to unstructured arrays
        >>> transform = tonic.transforms.NumpyAsType(int)
        >>> dataset = tonic.datasets.NMNIST(save_to='data', transform=transform)
        >>> dataloader = torch.utils.data.DataLoader(dataset)
        >>> events, targets = next(iter(dataloader))
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
        elif not source_is_structured_array and target_is_structured_array:
            return np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        elif source_is_structured_array and target_is_structured_array:
            return NotImplementedError
        elif not source_is_structured_array and not target_is_structured_array:
            return events.astype(self.dtype)
        else:
            raise ValueError("Something went wrong")


@dataclass(frozen=True)
class ToAveragedTimesurface:
    """Create averaged timesurfaces for each event. Taken from the paper Sironi et al. 2018, HATS:
    Histograms of averaged time surfaces for robust event-based object classification https://opena
    ccess.thecvf.com/content_cvpr_2018/papers/Sironi_HATS_Histograms_of_CVPR_2018_paper.pdf.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        cell_size (int): size of each square in the grid
        surface_size (int): has to be odd
        time_window (float): how far back to look for past events for the time averaging
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    """

    sensor_size: Tuple[int, int, int]
    surface_size: int = 5
    cell_size: int = 10
    time_window: float = 1e3
    tau: float = 100
    decay: str = "exp"

    def __call__(self, events):
        return functional.to_averaged_timesurface_numpy(
            events,
            sensor_size=self.sensor_size,
            cell_size=self.cell_size,
            surface_size=self.surface_size,
            time_window=self.time_window,
            tau=self.tau,
            decay=self.decay,
        )


@dataclass(frozen=True)
class ToFrame:
    """Accumulate events to frames by slicing along constant time (time_window), constant number of
    events (event_count) or constant number of frames (n_time_bins / n_event_bins). All the events
    in one slice are added up in a frame for each polarity.  If you want binary frames, you can
    manually clamp them to 1 afterwards. You can set one of the first 4 parameters to choose the
    slicing method. Depending on which method you choose, overlap will be defined differently. As a
    rule of thumb, here are some considerations if you are unsure which slicing method to choose:

    * If your recordings are of roughly the same length, a safe option is to set time_window. Bare in mind
      that the number of events can vary greatly from slice to slice, but will give you some consistency when
      training RNNs or other algorithms that have time steps.

    * If your recordings have roughly the same amount of activity / number of events and you are more interested
      in the spatial composition, then setting event_count will give you frames that are visually more consistent.

    * The previous time_window and event_count methods will likely result in a different amount of frames for each
      recording. If your training method benefits from consistent number of frames across a dataset (for easier
      batching for example), or you want a parameter that is easier to set than the exact window length or number
      of events per slice, consider fixing the number of frames by setting n_time_bins or n_event_bins. The two
      methods slightly differ with respect to how the slices are distributed across the recording. You can define
      an overlap between 0 and 1 to provide some robustness.

    Parameters:
        sensor_size: A 3-tuple of x,y,p for sensor_size. If omitted, the sensor size is calculated for that sample. However,
                    do use this feature sparingly as when not all pixels fire in a sample, this might cause issues with batching/
                    stacking tensors further down the line.
        time_window (float): Time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        event_count (int): Number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): Fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): Fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): Overlap between frames. The definition of overlap depends on the slicing method.
                         For slicing by time_window, the overlap is defined in microseconds. For slicing by event_count,
                         the overlap is defined by number of events. For slicing by n_time_bins or n_event_bins, the
                         overlap is defined by the fraction of a bin between 0 and 1.
        include_incomplete (bool): If True, includes overhang slice when time_window or event_count is specified.
                                   Not valid for bin_count methods.

    Example:
        >>> from tonic.transforms import ToFrame
        >>> transform1 = ToFrame(time_window=10000, overlap=1000, include_incomplete=True)
        >>> transform2 = ToFrame(event_count=3000, overlap=100, include_incomplete=True)
        >>> transform3 = ToFrame(n_time_bins=100, overlap=0.1)
    """

    sensor_size: Optional[Tuple[int, int, int]]
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
class ToSparseTensor:
    """PyTorch sparse tensor drop-in replacement for ToFrame. See
    https://pytorch.org/docs/stable/sparse.html for details about sparse tensors. The dense shape
    of the tensor will be (TCWH) and can be inflated by calling to_dense(). You need to have
    PyTorch installed for this transformation. Under the hood this transform calls ToFrame() with
    the same parameters, converts to a pytorch tensor and calls to_sparse().

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size. If omitted, the sensor size is calculated for that sample. However,
                    do use this feature sparingly as when not all pixels fire in a sample, this might cause issues with batching/
                    stacking tensors further down the line.
        time_window (float): time window length for one frame. Use the same time unit as timestamps in the event recordings.
                             Good if you want temporal consistency in your training, bad if you need some visual consistency
                             for every frame if the recording's activity is not consistent.
        event_count (int): number of events per frame. Good for training CNNs which do not care about temporal consistency.
        n_time_bins (int): fixed number of frames, sliced along time axis. Good for generating a pre-determined number of
                           frames which might help with batching.
        n_event_bins (int): fixed number of frames, sliced along number of events in the recording. Good for generating a
                            pre-determined number of frames which might help with batching.
        overlap (float): overlap between frames defined either in time units, number of events or number of bins between 0 and 1.
        include_incomplete (bool): if True, includes overhang slice when time_window or event_count is specified.
                                   Not valid for bin_count methods.

    Example:
        >>> from tonic.transforms import ToSparseTensor
        >>> transform1 = ToSparseTensor(time_window=10000, overlap=300, include_incomplete=True)
        >>> transform2 = ToSparseTensor(event_count=3000, overlap=100, include_incomplete=True)
        >>> transform3 = ToSparseTensor(n_time_bins=100, overlap=0.1)
    """

    sensor_size: Tuple[int, int, int]
    time_window: Optional[float] = None
    event_count: Optional[int] = None
    n_time_bins: Optional[int] = None
    n_event_bins: Optional[int] = None
    overlap: float = 0
    include_incomplete: bool = False

    def __call__(self, events):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed.")

        dense_frames = functional.to_frame_numpy(
            events=events,
            sensor_size=self.sensor_size,
            time_window=self.time_window,
            event_count=self.event_count,
            n_time_bins=self.n_time_bins,
            n_event_bins=self.n_event_bins,
            overlap=self.overlap,
            include_incomplete=self.include_incomplete,
        )
        return torch.from_numpy(dense_frames).to_sparse()


@dataclass(frozen=True)
class ToImage:
    """Counts up all events to a *single* image of size sensor_size.

    ToImage will typically be used in combination with SlicedDataset to cut a recording into
    smaller chunks that are then individually binned to frames.
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        frames = functional.to_frame_numpy(
            events=events, sensor_size=self.sensor_size, event_count=len(events)
        )

        return frames.squeeze(0)


@dataclass(frozen=True)
class ToTimesurface:
    """Create global time surfaces at a specific time interval dt.

    Parameters:
        sensor_size: A 3-tuple of x,y,p for sensor_size
        dt (float): The interval at which the time-surfaces are accumulated.
        tau (float): Time constant to decay events with.
    """

    sensor_size: Tuple[int, int, int]
    dt: float
    tau: float

    def __call__(self, events):
        return functional.to_timesurface_numpy(
            events=events,
            sensor_size=self.sensor_size,
            dt=self.dt,
            tau=self.tau,
        )


@dataclass(frozen=True)
class ToVoxelGrid:
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical
    flow, depth, and egomotion.

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        n_time_bins (int): fixed number of time bins to slice the event sample into.
    """

    sensor_size: Tuple[int, int, int]
    n_time_bins: int

    def __call__(self, events):
        return functional.to_voxel_grid_numpy(
            events.copy(), self.sensor_size, self.n_time_bins
        )


@dataclass(frozen=True)
class ToBinaRep:
    """Takes T*B binary event frames to produce a sequence of T frames of N-bit numbers. To do so,
    N binary frames are interpreted as a single frame of N-bit representation. Taken from the paper
    Barchid et al. 2022, Bina-Rep Event Frames: a Simple and Effective Representation for Event-
    based cameras https://arxiv.org/pdf/2202.13662.pdf.

    Parameters:
        n_frames (int): the number T of bina-rep frames.
        n_bits (int): the number N of bits used in the N-bit representation.


    Example:
        >>> n_time_bins = n_frames * n_bits
        >>>
        >>> transforms.Compose([
        >>>     transforms.ToFrame(
        >>>         sensor_size=sensor_size,
        >>>         n_time_bins=n_time_bins,
        >>>     ),
        >>>     transforms.ToBinaRep(
        >>>         n_frames=n_frames,
        >>>         n_bits=n_bits,
        >>>     ),
        >>> ])
    """

    n_frames: Optional[int] = 1
    n_bits: Optional[int] = 8

    def __call__(self, event_frames):
        return functional.to_bina_rep_numpy(event_frames, self.n_frames, self.n_bits)


@dataclass(frozen=True)
class Repeat:
    """Copies target n times.

    Useful to transform sample labels into sequences.
    """

    n_repeat: int

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.n_repeat, 1])


@dataclass(frozen=True)
class ToOneHotEncoding:
    """Transforms one or more targets into a one hot encoding scheme."""

    n_classes: int

    def __call__(self, target):
        return np.eye(self.n_classes)[target]
