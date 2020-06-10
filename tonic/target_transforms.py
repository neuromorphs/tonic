import numpy as np


class Compose(object):
    """Bundles several target transforms.

    Args:
        transforms (list of target transforms)
    """

    def __init__(self, target_transform):
        self.target_transform = target_transform

    def __call__(self, target):
        for tt in self.target_transform:
            target = tt(target=target)
        return target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.target_transform:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Repeat(object):
    """Copies target n times. Useful to transform sample labels into sequences."""

    def __init__(self, repetitions):
        self.repetitions = repetitions

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.repetitions, 1])


class ToOneHotEncoding(object):
    """Transforms one or more targets into a one hot encoding scheme."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, target):
        return np.eye(self.n_classes)[target]
