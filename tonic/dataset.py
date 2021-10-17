import os.path
from .download_utils import check_integrity, download_and_extract_archive
from pathlib import Path


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
        """
        Downloads from a given url, places into target folder and verifies the file hash.
        """
        download_and_extract_archive(
                self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
            )

    def _is_file_present(self):
        """
        Check if the dataset file (can be .zip, .rosbag, .hdf5,...) is present on disk. No hashing.
        """
        return check_integrity(os.path.join(self.location_on_system, self.filename))
    
    def _folder_contains_at_least_n_files_of_type(self, n_files: int, file_type: str) -> bool:
        """
        Check if the target folder `folder_name` contains at least a minimum amount of files, hinting that the 
        original archive has probably been extracted.
        """
        return len(list(Path(self.location_on_system, self.folder_name).glob(f"*/*{file_type}"))) >= n_files

    def _check_exists(self):
        """
        This function is supposed to do some lightweight checking to see if the downloaded files are 
        present and extracted if need be. Hashing all downloaded files takes too long for large datasets.
        """
        return NotImplementedError