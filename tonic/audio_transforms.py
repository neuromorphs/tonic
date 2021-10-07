from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class FixLength:
    """
    Fix the length of a sample along a specified axis to a given length.

    Parameters:
        length:
            Desired length of the sample
        axis:
            Dimension along which the length needs to be fixed.
    Args:
        data: torch.Tensor
    Returns:
        torch.Tensor of the same dimension
    """
    length: int
    axis: int = 1

    def __call__(self, data: torch.Tensor):
        data_length = data.shape[self.axis]
        if data_length == self.length:
            return data
        elif data_length > self.length:
            data_splits = torch.split(data, self.length, self.axis)
            return data_splits[0]
        else:
            padding = []
            for cur_axis, axis_len in enumerate(data.shape):
                if cur_axis == self.axis:
                    padding = [0, self.length - data_length] + padding
                else:
                    padding = [0, 0] + padding
            return F.pad(data, padding)


#@dataclass
#class BinNumpy:
#    orig_freq: float
#    new_freq: float
#
#    def __call__(self, data: np.ndarray):
#        x, y = np.where(data)
#        w = data[x, y]
#        subsampled_timesteps = np.linspace(0, data.shape[0], int(self.new_freq * data.shape[0] / self.orig_freq) + 1)
#        bins = (subsampled_timesteps, range(data.shape[1] + 1))
#        binned, _, _ = np.histogram2d(x, y, bins=bins, weights=w, density=False)
#        return binned


@dataclass
class Bin:
    """
    Bin the given data along a specified axis at the specivied new frequency

    Parameters:
        orig_freq: float
            Sampling frequency of the given data stream
        new_freq: float
            Desired frequency after binning
        axis: int
            Axis along which the data needs to be binned

    Args:
         data: torch.Tensor
            data to be binned

    Returns:
        torch.Tensor binned data

    """
    orig_freq: float
    new_freq: float
    axis: int

    def __call__(self, data: torch.Tensor):
        splits = torch.split(data, int(self.orig_freq/self.new_freq), dim=self.axis)
        data = [torch.sum(split, dim=self.axis, keepdim=True) for split in splits]
        return torch.cat(data, self.axis)

