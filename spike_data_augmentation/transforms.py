from . import functional


class Compose(object):
    """Bundles several transforms.

    Args:
        transforms (list of transforms)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, sensor_size, ordering):
        for t in self.transforms:
            events = t(events, sensor_size, ordering)
        return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TimeJitter(object):
    def __init__(self, variance=1):
        self.variance = variance

    def __call__(self, events, sensor_size, ordering):
        return functional.time_jitter_numpy(
            events, sensor_size, ordering, self.variance
        )


class SpatialJitter(object):
    def __init__(self, variance_x=1, variance_y=1, sigma_x_y=0):
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.sigma_x_y = sigma_x_y

    def __call__(self, events, sensor_size, ordering):
        return functional.spatial_jitter_numpy(
            events,
            sensor_size,
            ordering,
            self.variance_x,
            self.variance_y,
            self.sigma_x_y,
        )
