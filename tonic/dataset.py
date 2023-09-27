import os.path
from pathlib import Path
from typing import Callable, Optional

from .download_utils import check_integrity, download_and_extract_archive


class Dataset:
    """Base class for Tonic datasets which download public data.

    Contains a few helper function to reduce duplicated code.
    """

    def __init__(
        self,
        save_to: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        self.location_on_system = os.path.join(save_to, self.__class__.__name__)
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.data = []
        self.targets = []
        self.folder_name = ""

    def __repr__(self):
        return self.__class__.__name__

    def download(self) -> None:
        """Downloads from a given url, places into target folder and verifies the file hash."""
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )

    def _is_file_present(self) -> bool:
        """Check if the dataset file (can be .zip, .rosbag, .hdf5,...) is present on disk.

        No hashing.
        """
        return check_integrity(os.path.join(self.location_on_system, self.filename))

    def _folder_contains_at_least_n_files_of_type(
        self, n_files: int, file_type: str
    ) -> bool:
        """Check if the target folder `folder_name` contains at least a minimum amount of files,
        hinting that the original archive has probably been extracted."""
        return (
            len(
                list(
                    Path(self.location_on_system, self.folder_name).glob(
                        f"**/*{file_type}"
                    )
                )
            )
            >= n_files
        )

    def _check_exists(self):
        """This function is supposed to do some lightweight checking to see if the downloaded files
        are present and extracted if need be.

        Hashing all downloaded files takes too long for large datasets.
        """
        return NotImplementedError
