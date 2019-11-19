from . import functional


class Compose(object):
    """Bundles several transforms.

    Args:
        transforms (list of transforms)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        for t in self.transforms:
            events, images = t(
                events=events,
                images=images,
                sensor_size=sensor_size,
                ordering=ordering,
                multi_image=multi_image,
            )
        if multi_image and images.all() != None:
            return events, images
        else:
            return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Crop(object):
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.crop_numpy(
            events, images, sensor_size, ordering, self.target_size, multi_image
        )


class DropEvent(object):
    def __init__(self, drop_probability=0.5, random_drop_probability=False):
        self.drop_probability = drop_probability
        self.random_drop_probability = random_drop_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.drop_event_numpy(
            events, self.drop_probability, self.random_drop_probability
        )
        return events, images


class FlipLR(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability_lr = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_lr_numpy(
            events, images, sensor_size, ordering, self.flip_probability_lr, multi_image
        )


class FlipPolarity(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability_pol = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.flip_polarity_numpy(
            events, self.flip_probability_pol, ordering
        )
        return events, images


class FlipUD(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability_ud = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_ud_numpy(
            events, images, sensor_size, ordering, self.flip_probability_ud, multi_image
        )


class RefractoryPeriod(object):
    def __init__(self, refractory_period=0.5):
        self.refractory_period = refractory_period

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.refractory_period_numpy(
            events, sensor_size, ordering, self.refractory_period
        )
        return events, images


class SpatialJitter(object):
    def __init__(self, variance_x=1, variance_y=1, sigma_x_y=0):
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.sigma_x_y = sigma_x_y

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.spatial_jitter_numpy(
            events,
            sensor_size,
            ordering,
            self.variance_x,
            self.variance_y,
            self.sigma_x_y,
        )
        return events, images


class SpatioTemporalTransform(object):
    def __init__(self, spatial_transform, temporal_transform, roll=False):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.roll = roll

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.st_transform(
            events,
            sensor_size,
            ordering,
            self.spatial_transform,
            self.temporal_transform,
            self.roll,
        )
        return events, images


class TimeJitter(object):
    def __init__(self, variance=1):
        self.variance = variance

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_jitter_numpy(events, ordering, self.variance)
        return events, images


class TimeReversal(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability_t = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.time_reversal_numpy(
            events, images, sensor_size, ordering, self.flip_probability_t, multi_image
        )


class TimeSkew(object):
    def __init__(self, coefficient=0.9, offset=0):
        self.coefficient = coefficient
        self.offset = offset

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_skew_numpy(
            events, ordering, self.coefficient, self.offset
        )
        return events, images
