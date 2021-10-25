import os
import shutil
import glob
import numpy as np
import loris
import numpy
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class NavGesture(Dataset):
    """NavGesture dataset <https://www.neuromorphic-vision.com/public/downloads/navgesture/>. Events have (txyp) ordering.
    ::

        @article{maro2020event,
          title={Event-based gesture recognition with dynamic background suppression using smartphone computational capabilities},
          author={Maro, Jean-Matthieu and Ieng, Sio-Hoi and Benosman, Ryad},
          journal={Frontiers in neuroscience},
          volume={14},
          pages={275},
          year={2020},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        walk_subset (bool): Choose either NavGesture-sit (default) or NavGesture-walk dataset. No train/test split provided.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://www.neuromorphic-vision.com/public/downloads/navgesture/"
    sit_filename = "navgesture-sit.zip"
    walk_filename = "navgesture-walk.zip"
    sit_url = base_url + sit_filename
    walk_url = base_url + walk_filename
    sit_md5 = "1571753ace4d9e0946e6503313712c22"
    walk_md5 = "5d305266f13005401959e819abe206f0"

    classes = ["swipe down", "swipe up", "swipe left", "swipe right", "select", "home"]
    class_codes = ["do", "up", "le", "ri", "se", "ho"]
    int_classes = dict(zip(class_codes, range(len(class_codes))))
    sensor_size = (304, 240, 2)
    dtype = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")])
    ordering = dtype.names

    def __init__(
        self, save_to, walk_subset=False, transform=None, target_transform=None
    ):
        super(NavGesture, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.walk_subset = walk_subset

        if walk_subset:
            self.url = self.walk_url
            self.file_md5 = self.walk_md5
            self.filename = self.walk_filename
            self.folder_name = "navgesture-walk"
        else:
            self.url = self.sit_url
            self.file_md5 = self.sit_md5
            self.filename = self.sit_filename
            self.folder_name = "navgesture-sit"

        if not self._check_exists():
            self.download()
            data_folder = os.path.join(self.location_on_system, self.folder_name)
            # normally zips contain a top-level folder where we can extract to,
            # but here we have to create and move the data into it manually
            os.makedirs(data_folder, exist_ok=True)
            pattern = "/user*.zip"
            files = glob.glob(self.location_on_system + pattern)
            for file in files:
                file_name = os.path.basename(file)
                shutil.move(file, os.path.join(data_folder, file_name))
            for path, dirs, files in os.walk(data_folder):
                dirs.sort()
                for file in files:
                    if file.startswith("user") and file.endswith("zip"):
                        extract_archive(os.path.join(data_folder, file))

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("dat"):
                    self.data.append(path + "/" + file)
                    self.targets.append(self.int_classes[file[7:9]])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events, target = (
            loris.read_file(self.data[index])["events"],
            self.targets[index],
        )
        events = np.lib.recfunctions.rename_fields(events, {'ts': 't', 'is_increase': 'p'})

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            304, ".dat"
        )
