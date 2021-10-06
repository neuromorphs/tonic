from dataclasses import dataclass

import torch
import torch.nn.functional as F

@dataclass
class StandardizeDataLength:
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
            new_shape = data.shape()
            new_shape[self.axis] = self.length
            zeros = torch.zeros(new_shape)
            ...
            F.pad(data, (0, self.length-data_length), )
