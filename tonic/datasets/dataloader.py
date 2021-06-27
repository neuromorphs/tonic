import numpy as np


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
        return self

    def __next__(self):
        if self.iteration >= self.length:
            raise StopIteration
        elif self.batch_size == 1:
            data = self.dataset[self.indices[self.iteration]]
            self.iteration += 1
            return data
        else:
            begin = self.iteration
            end = self.iteration + self.batch_size
            data = self.dataset[self.indices[begin:end]]
            self.iteration += len(data)
            return data

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))