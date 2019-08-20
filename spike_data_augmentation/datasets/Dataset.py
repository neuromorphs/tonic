class Dataset:
    def __init__(self, save_to="./", transform=None):
        self.location_on_system = save_to
        self.transform = transform
        self.data = []
        self.targets = []

    def __repr__(self):
        return "Dataset " + self.__class__.__name__
