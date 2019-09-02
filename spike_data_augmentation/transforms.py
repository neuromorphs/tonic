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
    def __init__(self, images=None, target_size=(256, 256), multi_image=None):
        self.images = images
        self.target_size = target_size
        self.multi_image = multi_image

    def __call__(self, events, sensor_size, ordering):
        return functional.crop_numpy(
            events,
            self.images,
            sensor_size,
            ordering,
            self.target_size,
            self.multi_image,
        )


class FlipLR(object):
    def __init__(self, images=None, flip_probability=0.5, multi_image=None):
        self.images = images
        self.flip_probability = flip_probability
        self.multi_image = multi_image

    def __call__(self, events, sensor_size, ordering):
        return functional.flip_lr_numpy(
            events,
            self.images,
            sensor_size,
            ordering,
            self.flip_probability,
            self.multi_image,
        )


class FlipUD(object):
    def __init__(self, images=None, flip_probability=0.5, multi_image=None):
        self.images = images
        self.flip_probability = flip_probability
        self.multi_image = multi_image

    def __call__(self, events, sensor_size, ordering):
        return functional.flip_ud_numpy(
            events,
            self.images,
            sensor_size,
            ordering,
            self.flip_probability,
            self.multi_image,
        )


class DropEvent(object):
    def __init__(self, drop_probability=0.5, random_drop_probability=False):
        self.drop_probability = drop_probability
        self.random_drop_probability = random_drop_probability

    def __call__(self, events, sensor_size, ordering):
        return functional.drop_event_numpy(
            events,
            sensor_size,
            ordering,
            self.drop_probability,
            self.random_drop_probability,
        )


class RefractoryPeriod(object):
    def __init__(self, refractory_period=0.5):
        self.refractory_period = refractory_period

    def __call__(self, events, sensor_size, ordering):
        return functional.refractory_period_numpy(
            events, sensor_size, ordering, self.refractory_period
        )


class SpatialJitter(object):
    def __init__(self, variance_x=1, variance_y=1, sigma_x_y=0):
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.sigma_x_y = sigma_x_y

    def __call__(self, events, sensor_size, ordering, images=None):
        events = functional.spatial_jitter_numpy(
            events, ordering, self.variance_x, self.variance_y, self.sigma_x_y
        )
        return (events, None)


class SpatioTemporalTransform(object):
    def __init__(self, spatial_transform, temporal_transform, roll=False):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.roll = roll

    def __call__(self, events, sensor_size, ordering):
        return functional.st_transform(
            events,
            sensor_size,
            ordering,
            self.spatial_transform,
            self.temporal_transform,
            self.roll,
        )


class TimeJitter(object):
    def __init__(self, variance=1):
        self.variance = variance

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_jitter_numpy(events, ordering, self.variance)
        return (events, images)


class TimeReversal(object):
    def __init__(self, flip_probability=0.5):
        self.flip_probability = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.time_reversal_numpy(
            events, images, sensor_size, ordering, self.flip_probability, multi_image
        )


class TimeSkew(object):
    def __init__(self, coefficient=0.9, offset=0):
        self.coefficient = coefficient
        self.offset = offset

    def __call__(self, events, sensor_size, ordering):
        return functional.time_skew_numpy(
            events, ordering, self.coefficient, self.offset
        )
