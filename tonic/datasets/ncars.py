import os
import loris
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class NCARS(Dataset):
    """N-Cars dataset <https://www.prophesee.ai/dataset-n-cars-download/>. Events have (txyp) ordering.
    ::

        @inproceedings{sironi2018hats,
          title={HATS: Histograms of averaged time surfaces for robust event-based object classification},
          author={Sironi, Amos and Brambilla, Manuele and Bourdis, Nicolas and Lagorce, Xavier and Benosman, Ryad},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1731--1740},
          year={2018}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "http://www.prophesee.ai/resources/Prophesee_Dataset_n_cars.zip"
    filename = "Prophesee_Dataset_n_cars.zip"
    train_filename = "n-cars_train.zip"
    test_filename = "n-cars_test.zip"
    file_md5 = "553ce464d6e5e617b3c21ce27c19368e"
    classes = ["background", "car"]

    class_dict = {"background": 0, "cars": 1}

    sensor_size = None  # different for every recording
    minimum_y_value = 140
    dtype = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")])
    ordering = "txyp"

    def __init__(self, save_to, train=True, transform=None, target_transform=None):
        super(NCARS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if train:
            self.folder_name = "train"
        else:
            self.folder_name = "test"

        if not self._check_exists():
            self.download()
            extract_archive(os.path.join(self.location_on_system, self.train_filename))
            extract_archive(os.path.join(self.location_on_system, self.test_filename))

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("dat"):
                    self.data.append(path + "/" + file)
                    self.targets.append(self.class_dict[os.path.basename(path)])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = loris.read_file(self.data[index])["events"]
        events = np.lib.recfunctions.rename_fields(events, {'ts': 't', 'is_increase': 'p'})
        events["y"] -= self.minimum_y_value
        events["y"] = events["y"].max() - events["y"]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            8000, ".dat"
        )
