import numpy as np

from copy import copy

class DataLoader:
    def __init__(self, dataset, shuffle=False, batch_size=1):
        self.dataset = dataset
        self.length = len(dataset)
        self.indices = np.arange(0, self.length, dtype=int)
        self.iteration = 0
        self.batch_size = batch_size
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        # Return a copy of data loader so state won't be modified
        return copy(self)

    def __next__(self):
        if self.iteration >= self.length:
            raise StopIteration
        elif self.batch_size == 1:
            data = self.dataset[self.indices[self.iteration]]
            self.iteration += 1
            return data
        else:
            # Get start and end of batch (end might be past end of indices)
            begin = self.iteration
            end = self.iteration + self.batch_size
            # Get indices and thus slice of data
            inds = self.indices[begin:end]
            data = self.dataset[inds]
            # Add number of indices to iteration count
            # (will take into account size of dataset)
            self.iteration += len(inds)
            return data

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))