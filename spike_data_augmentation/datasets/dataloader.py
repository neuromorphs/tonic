import numpy as np


class Dataloader:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.indices = np.arange(0, len(dataset))
        self.iteration = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        self.iteration = self.iteration + 1
        if self.iteration >= len(self.indices):
            raise StopIteration
        else:
            return self.dataset[self.indices[self.iteration]]

    def __len__(self):
        return len(self.dataset)
