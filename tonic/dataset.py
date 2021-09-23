import os.path
from .download_utils import check_integrity, download_and_extract_archive


class Dataset:
    def __init__(self, save_to="./", transform=None, target_transform=None):
        self.location_on_system = save_to
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

    def __repr__(self):
        return "Dataset " + self.__class__.__name__

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )

    def check_integrity(self):
        root = self.location_on_system
        fpath = os.path.join(root, self.filename)

        if not check_integrity(fpath, self.file_md5):
            return False
        return True
