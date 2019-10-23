import numpy as np


class Dataloader:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.length = len(dataset)
        self.indices = np.arange(0, self.length)
        self.iteration = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.length:
            raise StopIteration
        else:
            data = self.dataset[self.indices[self.iteration]]
            self.iteration += 1
            return data

    def __len__(self):
        return len(self.dataset)
