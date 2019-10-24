class Dataset:
    def __init__(self, save_to="./", transform=None, representation=None):
        self.location_on_system = save_to
        self.transform = transform
        self.representation = representation
        self.data = []
        self.targets = []

    def __repr__(self):
        return "Dataset " + self.__class__.__name__

    def total_number_of_events(self):
        if self.data == []:
            return None
        else:
            total_number_of_events = 0
            for recording in self.data:
                total_number_of_events += len(recording)
            return total_number_of_events
